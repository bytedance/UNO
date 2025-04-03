# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from pycocotools import mask as coco_mask
from torchvision.transforms import CenterCrop, Compose, Normalize, RandomResizedCrop, ToTensor
from torchvision.transforms.functional import InterpolationMode

cv2.setNumThreads(0)


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


# Subject Dataset
class SubjectMultiDataset(torch.utils.data.Dataset):
    def __init__(self, data_root, args=None, samples=None):
        super().__init__()
        self.args = args

        # read dataset json (may contation multi-datasets)
        self.transform = Compose([
            ToTensor(),
            Normalize([0.5], [0.5]),
        ])

        with open(data_root, 'r') as f:
            self.data = json.load(f)
        if samples is not None:
            self.data = samples    

    def _horizontal_concat(self, images):
        widths, heights = zip(*(img.size for img in images))
        
        total_width = sum(widths)
        max_height = max(heights)
        
        new_im = Image.new('RGB', (total_width, max_height))
        
        x_offset = 0
        for img in images:
            new_im.paste(img, (x_offset, 0))
            x_offset += img.size[0]
        
        return new_im

    def _process_image_varlen(self, image, predefined_scales=None, bbox=None, mask_rle=None, cropped_ratio=None):
        if mask_rle is not None:
            raw_image = cv2.imread(image)
            mask_decode = coco_mask.decode(mask_rle)
            mask = cv2.cvtColor(mask_decode, cv2.COLOR_GRAY2BGR)
            raw_image = raw_image * mask

            # 创建一个纯白色背景图片
            mask_reverse = (mask==0)*255
            raw_image = raw_image + mask_reverse

            raw_image = Image.fromarray(cv2.cvtColor(raw_image.astype(np.uint8),cv2.COLOR_BGR2RGB)) 
        else:
            raw_image = Image.open(image)
        if cropped_ratio is not None:
            # 防止白边出现，先中心crop一下
            w, h = raw_image.size
            crop_w, crop_h = (1-cropped_ratio)*w, (1-cropped_ratio)*h
            crop = CenterCrop((int(crop_h), int(crop_w)))
            raw_image = crop(raw_image)
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            img_w, img_h = raw_image.size
            if mask_rle is not None:
                # 中心不变，四周按比例膨胀框大小
                xc, yc = (x1+x2)/2, (y1+y2)/2
                w, h = x2-x1, y2-y1
                x1, y1 = xc-w/2 * 1.2, yc-h/2 * 1.2
                x2, y2 = xc+w/2 * 1.2, yc+h/2 * 1.2
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(img_w, x2)
                y2 = min(img_h, y2)
            raw_image = raw_image.crop((x1, y1, x2, y2))
        crop_coords_top_left = None
        if predefined_scales is not None:
            image_w, image_h = raw_image.size
            # 为了varlen inference，把图片按bucket进行resize
            bucket_h, bucket_w = find_nearest_scale(image_h, image_w, predefined_scales)
    
            aspect_ratio = bucket_w / bucket_h
            resize = RandomResizedCrop(
                                        size=(bucket_h, bucket_w),  # Crop to target width height
                                        scale=(1, 1),  # Do not scale.
                                        ratio=(aspect_ratio, aspect_ratio),  # Keep target aspect ratio.
                                        interpolation=InterpolationMode.LANCZOS  # Use LANCZO for downsample.
                                    )
            crop_top_coord, crop_left_coord, _, _ = resize.get_params(raw_image, scale=(1, 1), ratio=(
                                aspect_ratio, aspect_ratio))
            crop_coords_top_left = torch.tensor([crop_top_coord, crop_left_coord])
            raw_image = resize(raw_image)
        raw_image = raw_image.convert("RGB")            
        
        return raw_image, crop_coords_top_left


    def __getitem__(self, idx): 
        sample = self.data[idx]
        comb_idx = sample["comb_idx"]
        item = sample["index"]
        prompt = sample["prompt"]

        entity1_img_paths = sample['img_path1']
        entity2_img_paths = sample['img_path2']

        predefined_scales=[
            # w    h
            (320, 256),
            (384, 256),        
            (320, 320),
            (256, 320),        
            (256, 384),
        ]      
        # read and process image
        entity1_imgs_raw = self._process_image_varlen(entity1_img_paths, predefined_scales)[0]
        entity2_imgs_raw = self._process_image_varlen(entity2_img_paths, predefined_scales)[0]

        entity1_imgs = self.transform(entity1_imgs_raw)
        entity2_imgs = self.transform(entity2_imgs_raw) 

        if self.args.rank == 1:
            ## 存图片 ##
            save_concat_img = self._horizontal_concat([entity1_imgs_raw, entity2_imgs_raw])
            # 创建一个可以在图像上绘图的对象
            draw = ImageDraw.Draw(save_concat_img) 
            # 选择字体和大小
            font = ImageFont.truetype("DejaVuSans.ttf", 10)

            # 在图像上绘制文本
            draw.text((8,8), prompt, font=font, fill=(255, 0, 0))
            save_ref_dir = os.path.join(self.args.save_path, 'ref_imgs')
            os.makedirs(save_ref_dir, exist_ok=True)
            save_concat_img.save(os.path.join(save_ref_dir, f"ref_{comb_idx}_{item}.png"))
            ## 存图片 ##          

        ret_val = {
                   "ref_img1": entity1_imgs, 
                   "ref_img2": entity2_imgs,
                   "txt": prompt,
                   "comb_idx": comb_idx,
                   "index":f"{comb_idx}_{item}"}
        return ret_val

    def __len__(self):
        return len(self.data)
       

def get_dreambench_multiip_dataloader(args, samples=None):
    dataset = SubjectMultiDataset(args.eval_data_json, args, samples=samples)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=2, prefetch_factor=4, pin_memory=True)
    return dataloader