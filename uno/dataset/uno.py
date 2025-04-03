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

import json
import os

import numpy as np
import torch
import torchvision.transforms.functional as TVF
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, Normalize, ToTensor

def bucket_images(images: list[torch.Tensor], resolution: int = 512):
    bucket_override=[
        # h    w
        (256, 768),
        (320, 768),
        (320, 704),
        (384, 640),
        (448, 576),
        (512, 512),
        (576, 448),
        (640, 384),
        (704, 320),
        (768, 320),
        (768, 256)
    ]
    bucket_override = [(int(h / 512 * resolution), int(w / 512 * resolution)) for h, w in bucket_override]
    bucket_override = [(h // 16 * 16, w // 16 * 16) for h, w in bucket_override]

    aspect_ratios = [image.shape[-2] / image.shape[-1] for image in images]
    mean_aspect_ratio = np.mean(aspect_ratios)
    
    new_h, new_w = bucket_override[0]
    min_aspect_diff = np.abs(new_h / new_w - mean_aspect_ratio)
    for h, w in bucket_override:
        aspect_diff = np.abs(h / w - mean_aspect_ratio)
        if aspect_diff < min_aspect_diff:
            min_aspect_diff = aspect_diff
            new_h, new_w = h, w
    
    images = [TVF.resize(image, (new_h, new_w)) for image in images]
    images = torch.stack(images, dim=0)
    return images

class FluxPairedDatasetV2(Dataset):
    def __init__(self, json_file: str, resolution: int, resolution_ref: int | None = None):
        super().__init__()
        self.json_file = json_file
        self.resolution = resolution
        self.resolution_ref = resolution_ref if resolution_ref is not None else resolution
        self.image_root = os.path.dirname(json_file)

        with open(self.json_file, "rt") as f:
            self.data_dicts = json.load(f)
        
        self.transform = Compose([
            ToTensor(),
            Normalize([0.5], [0.5]),
        ])
    
    def __getitem__(self, idx):
        data_dict = self.data_dicts[idx]
        image_paths = [data_dict["image_path"]] if "image_path" in data_dict else data_dict["image_paths"]
        txt = data_dict["prompt"]
        image_tgt_path = data_dict.get("image_tgt_path", None)
        ref_imgs = [
            Image.open(os.path.join(self.image_root, path)).convert("RGB")
            for path in image_paths
        ]
        ref_imgs = [self.transform(img) for img in ref_imgs]
        img = None
        if image_tgt_path is not None:
            img = Image.open(os.path.join(self.image_root, image_tgt_path)).convert("RGB")
            img = self.transform(img)

        return {
            "img": img,
            "txt": txt,
            "ref_imgs": ref_imgs,
        }

    def __len__(self):
        return len(self.data_dicts)
    
    def collate_fn(self, batch):
        img = [data["img"] for data in batch]
        txt = [data["txt"] for data in batch]
        ref_imgs = [data["ref_imgs"] for data in batch]
        assert all([len(ref_imgs[0]) == len(ref_imgs[i]) for i in range(len(ref_imgs))])

        n_ref = len(ref_imgs[0])

        img = bucket_images(img, self.resolution)
        ref_imgs_new = []
        for i in range(n_ref):
            ref_imgs_i = [refs[i] for refs in ref_imgs]
            ref_imgs_i = bucket_images(ref_imgs_i, self.resolution_ref)
            ref_imgs_new.append(ref_imgs_i)

        return {
            "txt": txt,
            "img": img,
            "ref_imgs": ref_imgs_new,
        }

if __name__ == '__main__':
    import argparse
    from pprint import pprint
    parser = argparse.ArgumentParser()
    # parser.add_argument("--json_file", type=str, required=True)
    parser.add_argument("--json_file", type=str, default="datasets/fake_train_data.json")
    args = parser.parse_args()
    dataset = FluxPairedDatasetV2(args.json_file, 512)
    dataloder = DataLoader(dataset, batch_size=4, collate_fn=dataset.collate_fn)

    for i, data_dict in enumerate(dataloder):
        pprint(i)
        pprint(data_dict)
        breakpoint()
