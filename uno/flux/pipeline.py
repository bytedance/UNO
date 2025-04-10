# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates. All rights reserved.
# Copyright (c) 2024 Black Forest Labs and The XLabs-AI Team. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from typing import Literal

import torch
from einops import rearrange
from PIL import ExifTags, Image
import torchvision.transforms.functional as TVF

from uno.flux.modules.layers import (
    DoubleStreamBlockLoraProcessor,
    DoubleStreamBlockProcessor,
    SingleStreamBlockLoraProcessor,
    SingleStreamBlockProcessor,
)
from uno.flux.sampling import denoise, get_noise, get_schedule, prepare_multi_ip, unpack
from uno.flux.util import (
    get_lora_rank,
    load_ae,
    load_checkpoint,
    load_clip,
    load_flow_model,
    load_flow_model_only_lora,
    load_flow_model_quintized,
    load_t5,
)


def find_nearest_scale(image_h, image_w, predefined_scales):
    """
    根据图片的高度和宽度，找到最近的预定义尺度。

    :param image_h: 图片的高度
    :param image_w: 图片的宽度
    :param predefined_scales: 预定义尺度列表 [(h1, w1), (h2, w2), ...]
    :return: 最近的预定义尺度 (h, w)
    """
    # 计算输入图片的长宽比
    image_ratio = image_h / image_w

    # 初始化变量以存储最小差异和最近的尺度
    min_diff = float('inf')
    nearest_scale = None

    # 遍历所有预定义尺度，找到与输入图片长宽比最接近的尺度
    for scale_h, scale_w in predefined_scales:
        predefined_ratio = scale_h / scale_w
        diff = abs(predefined_ratio - image_ratio)

        if diff < min_diff:
            min_diff = diff
            nearest_scale = (scale_h, scale_w)

    return nearest_scale

def preprocess_ref(raw_image: Image.Image, long_size: int = 512):
    # 获取原始图像的宽度和高度
    image_w, image_h = raw_image.size

    # 计算长边和短边
    if image_w >= image_h:
        new_w = long_size
        new_h = int((long_size / image_w) * image_h)
    else:
        new_h = long_size
        new_w = int((long_size / image_h) * image_w)

    # 按新的宽高进行等比例缩放
    raw_image = raw_image.resize((new_w, new_h), resample=Image.LANCZOS)
    target_w = new_w // 16 * 16
    target_h = new_h // 16 * 16

    # 计算裁剪的起始坐标以实现中心裁剪
    left = (new_w - target_w) // 2
    top = (new_h - target_h) // 2
    right = left + target_w
    bottom = top + target_h

    # 进行中心裁剪
    raw_image = raw_image.crop((left, top, right, bottom))

    # 转换为 RGB 模式
    raw_image = raw_image.convert("RGB")
    return raw_image

class UNOPipeline:
    def __init__(
        self,
        model_type: str,
        device: torch.device,
        offload: bool = False,
        only_lora: bool = False,
        lora_rank: int = 16
    ):
        self.device = device
        self.offload = offload
        self.model_type = model_type

        self.clip = load_clip(self.device)
        self.t5 = load_t5(self.device, max_length=512)
        self.ae = load_ae(model_type, device="cpu" if offload else self.device)
        self.use_fp8 = "fp8" in model_type
        if only_lora:
            self.model = load_flow_model_only_lora(
                model_type,
                device="cpu" if offload else self.device,
                lora_rank=lora_rank,
                use_fp8=self.use_fp8
            )
        else:
            self.model = load_flow_model(model_type, device="cpu" if offload else self.device)


    def load_ckpt(self, ckpt_path):
        if ckpt_path is not None:
            from safetensors.torch import load_file as load_sft
            print("Loading checkpoint to replace old keys")
            # load_sft doesn't support torch.device
            if ckpt_path.endswith('safetensors'):
                sd = load_sft(ckpt_path, device='cpu')
                missing, unexpected = self.model.load_state_dict(sd, strict=False, assign=True)
            else:
                dit_state = torch.load(ckpt_path, map_location='cpu')
                sd = {}
                for k in dit_state.keys():
                    sd[k.replace('module.','')] = dit_state[k]
                missing, unexpected = self.model.load_state_dict(sd, strict=False, assign=True)
                self.model.to(str(self.device))
            print(f"missing keys: {missing}\n\n\n\n\nunexpected keys: {unexpected}")

    def set_lora(self, local_path: str = None, repo_id: str = None,
                 name: str = None, lora_weight: int = 0.7):
        checkpoint = load_checkpoint(local_path, repo_id, name)
        self.update_model_with_lora(checkpoint, lora_weight)

    def set_lora_from_collection(self, lora_type: str = "realism", lora_weight: int = 0.7):
        checkpoint = load_checkpoint(
            None, self.hf_lora_collection, self.lora_types_to_names[lora_type]
        )
        self.update_model_with_lora(checkpoint, lora_weight)

    def update_model_with_lora(self, checkpoint, lora_weight):
        rank = get_lora_rank(checkpoint)
        lora_attn_procs = {}

        for name, _ in self.model.attn_processors.items():
            lora_state_dict = {}
            for k in checkpoint.keys():
                if name in k:
                    lora_state_dict[k[len(name) + 1:]] = checkpoint[k] * lora_weight

            if len(lora_state_dict):
                if name.startswith("single_blocks"):
                    lora_attn_procs[name] = SingleStreamBlockLoraProcessor(dim=3072, rank=rank)
                else:
                    lora_attn_procs[name] = DoubleStreamBlockLoraProcessor(dim=3072, rank=rank)
                lora_attn_procs[name].load_state_dict(lora_state_dict)
                lora_attn_procs[name].to(self.device)
            else:
                if name.startswith("single_blocks"):
                    lora_attn_procs[name] = SingleStreamBlockProcessor()
                else:
                    lora_attn_procs[name] = DoubleStreamBlockProcessor()

        self.model.set_attn_processor(lora_attn_procs)


    def __call__(
        self,
        prompt: str,
        width: int = 512,
        height: int = 512,
        guidance: float = 4,
        num_steps: int = 50,
        seed: int = 123456789,
        **kwargs
    ):
        width = 16 * (width // 16)
        height = 16 * (height // 16)

        device_type = self.device if isinstance(self.device, str) else self.device.type
        with torch.autocast(enabled=self.use_fp8, device_type=device_type, dtype=torch.bfloat16):
            return self.forward(
                prompt,
                width,
                height,
                guidance,
                num_steps,
                seed,
                **kwargs
            )

    @torch.inference_mode()
    def gradio_generate(
        self,
        prompt: str,
        width: int,
        height: int,
        guidance: float,
        num_steps: int,
        seed: int,
        image_prompt1: Image.Image,
        image_prompt2: Image.Image,
        image_prompt3: Image.Image,
        image_prompt4: Image.Image,
    ):
        ref_imgs = [image_prompt1, image_prompt2, image_prompt3, image_prompt4]
        ref_imgs = [img for img in ref_imgs if isinstance(img, Image.Image)]
        ref_long_side = 512 if len(ref_imgs) <= 1 else 320
        ref_imgs = [preprocess_ref(img, ref_long_side) for img in ref_imgs]

        seed = seed if seed != -1 else torch.randint(0, 10 ** 8, (1,)).item()

        img = self(prompt=prompt, width=width, height=height, guidance=guidance,
                   num_steps=num_steps, seed=seed, ref_imgs=ref_imgs)

        filename = f"output/gradio/{seed}_{prompt[:20]}.png"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        exif_data = Image.Exif()
        exif_data[ExifTags.Base.Make] = "UNO"
        exif_data[ExifTags.Base.Model] = self.model_type
        info = f"{prompt=}, {seed=}, {width=}, {height=}, {guidance=}, {num_steps=}"
        exif_data[ExifTags.Base.ImageDescription] = info
        img.save(filename, format="png", exif=exif_data)
        return img, filename

    @torch.inference_mode
    def forward(
        self,
        prompt: str,
        width: int,
        height: int,
        guidance: float,
        num_steps: int,
        seed: int,
        ref_imgs: list[Image.Image] | None = None,
        pe: Literal['d', 'h', 'w', 'o'] = 'd',
    ):
        x = get_noise(
            1, height, width, device=self.device,
            dtype=torch.bfloat16, seed=seed
        )
        timesteps = get_schedule(
            num_steps,
            (width // 8) * (height // 8) // (16 * 16),
            shift=True,
        )
        if self.offload:
            self.ae.encoder = self.ae.encoder.to(self.device)
        x_1_refs = [
            self.ae.encode(
                (TVF.to_tensor(ref_img) * 2.0 - 1.0) 
                .unsqueeze(0).to(self.device, torch.float32)
            ).to(torch.bfloat16)
            for ref_img in ref_imgs
        ]

        if self.offload:
            self.offload_model_to_cpu(self.ae.encoder)
            self.t5, self.clip = self.t5.to(self.device), self.clip.to(self.device)
        inp_cond = prepare_multi_ip(
            t5=self.t5, clip=self.clip,
            img=x,
            prompt=prompt, ref_imgs=x_1_refs, pe=pe
        )

        if self.offload:
            self.offload_model_to_cpu(self.t5, self.clip)
            self.model = self.model.to(self.device)

        x = denoise(
            self.model,
            **inp_cond,
            timesteps=timesteps,
            guidance=guidance,
        )

        if self.offload:
            self.offload_model_to_cpu(self.model)
            self.ae.decoder.to(x.device)
        x = unpack(x.float(), height, width)
        x = self.ae.decode(x)
        self.offload_model_to_cpu(self.ae.decoder)

        x1 = x.clamp(-1, 1)
        x1 = rearrange(x1[-1], "c h w -> h w c")
        output_img = Image.fromarray((127.5 * (x1 + 1.0)).cpu().byte().numpy())
        return output_img

    def offload_model_to_cpu(self, *models):
        if not self.offload: return
        for model in models:
            model.cpu()
            torch.cuda.empty_cache()
