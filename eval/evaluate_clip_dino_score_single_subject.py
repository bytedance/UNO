import argparse
import glob
import json
import os
import warnings
from pathlib import Path

import clip
import numpy as np
import pandas as pd
import sklearn.preprocessing
import torch
from packaging import version
from PIL import Image
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor
from tqdm import tqdm

# New added
import pickle


class CLIPCapDataset(torch.utils.data.Dataset):
    def __init__(self, data, append=False, prefix='A photo depicts'):
        self.data = data
        self.prefix = ''
        if append:
            self.prefix = prefix
            if self.prefix[-1] != ' ':
                self.prefix += ' '

    def __getitem__(self, idx):
        c_data = self.data[idx]
        c_data = clip.tokenize(self.prefix + c_data, truncate=True).squeeze()
        return {'caption': c_data}

    def __len__(self):
        return len(self.data)


def Convert(image):
    return image.convert("RGB")


class CLIPImageDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data
        # only 224x224 ViT-B/32 supported for now
        self.preprocess = self._transform_test(224)

    def _transform_test(self, n_px):
        return Compose([
            Resize(n_px, interpolation=Image.BICUBIC),
            CenterCrop(n_px),
            Convert,
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073),
                      (0.26862954, 0.26130258, 0.27577711)),
        ])

    def __getitem__(self, idx):
        c_data = self.data[idx]
        image = Image.open(c_data)
        image = self.preprocess(image)
        return {'image': image}

    def __len__(self):
        return len(self.data)


class DINOImageDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data
        # only 224x224 ViT-B/32 supported for now
        self.preprocess = self._transform_test(224)

    def _transform_test(self, n_px):
        return Compose([
            Resize(256, interpolation=Image.BICUBIC),
            CenterCrop(n_px),
            Convert,
            ToTensor(),
            Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def __getitem__(self, idx):
        c_data = self.data[idx]
        image = Image.open(c_data)
        image = self.preprocess(image)
        return {'image': image}

    def __len__(self):
        return len(self.data)


def extract_all_captions(captions, model, device, batch_size=256, num_workers=8, append=False):
    data = torch.utils.data.DataLoader(
        CLIPCapDataset(captions, append=append),
        batch_size=batch_size, num_workers=num_workers, shuffle=False)
    all_text_features = []
    with torch.no_grad():
        # for b in tqdm(data):
        for b in data:
            b = b['caption'].to(device)
            all_text_features.append(model.encode_text(b).cpu().numpy())
    all_text_features = np.vstack(all_text_features)
    return all_text_features


def extract_all_images(images, model, datasetclass, device, batch_size=64, num_workers=8):
    data = torch.utils.data.DataLoader(
        datasetclass(images),
        batch_size=batch_size, num_workers=num_workers, shuffle=False)
    all_image_features = []
    with torch.no_grad():
        # for b in tqdm(data):
        for b in data:
            b = b['image'].to(device)
            if hasattr(model, 'encode_image'):
                if device == 'cuda':
                    b = b.to(torch.float16)
                all_image_features.append(model.encode_image(b).cpu().numpy())
            else:
                all_image_features.append(model(b).cpu().numpy())
    all_image_features = np.vstack(all_image_features)
    return all_image_features


def get_clip_score(model, images, candidates, device, append=False, w=1.0):
    '''
    get standard image-text clipscore.
    images can either be:
    - a list of strings specifying filepaths for images
    - a precomputed, ordered matrix of image features
    '''
    if isinstance(images, list):
        # need to extract image features
        images = extract_all_images(images, model, device)

    candidates = extract_all_captions(candidates, model, device, append=append)

    if version.parse(np.__version__) < version.parse('1.21'):
        images = sklearn.preprocessing.normalize(images, axis=1)
        candidates = sklearn.preprocessing.normalize(candidates, axis=1)
    else:
        images = images / np.sqrt(np.sum(images ** 2, axis=1, keepdims=True))
        candidates = candidates / \
            np.sqrt(np.sum(candidates ** 2, axis=1, keepdims=True))

    per = w * np.clip(np.sum(images * candidates, axis=1), 0, None)
    return np.mean(per), per


def clipeval(image_paths, prompts, model, device):
    image_feats = extract_all_images(
        image_paths, model, CLIPImageDataset, device, batch_size=64, num_workers=8)

    _, per_instance_image_text = get_clip_score(
        model, image_feats, prompts, device)

    scores = {image_path: {'CLIPScore': float(clipscore)}
              for image_path, clipscore in
              zip(image_paths, per_instance_image_text)}

    return np.mean([s['CLIPScore'] for s in scores.values()]), np.std([s['CLIPScore'] for s in scores.values()])

def clipeval_image(image_paths, image_dir_ref, model, device):
    image_paths_ref = [os.path.join(image_dir_ref, path) for path in os.listdir(image_dir_ref)
                       if path.endswith(('.png', '.jpg', '.jpeg', '.tiff', '.JPG'))]

    image_feats = extract_all_images(
        image_paths, model, CLIPImageDataset, device, batch_size=64, num_workers=8)
    image_feats_ref = extract_all_images(
        image_paths_ref, model, CLIPImageDataset, device, batch_size=64, num_workers=8)
    image_feats = image_feats / \
        np.sqrt(np.sum(image_feats ** 2, axis=1, keepdims=True))
    image_feats_ref = image_feats_ref / \
        np.sqrt(np.sum(image_feats_ref ** 2, axis=1, keepdims=True))
    res = image_feats @ image_feats_ref.T
    return np.mean(res)

def dinoeval_image(image_paths, image_dir_ref, model, device):
    image_paths_ref = [os.path.join(image_dir_ref, path) for path in os.listdir(image_dir_ref)
                       if path.endswith(('.png', '.jpg', '.jpeg', '.tiff', '.JPG'))]

    image_feats = extract_all_images(
        image_paths, model, DINOImageDataset, device, batch_size=64, num_workers=8)

    image_feats_ref = extract_all_images(
        image_paths_ref, model, DINOImageDataset, device, batch_size=64, num_workers=8)

    image_feats = image_feats / \
        np.sqrt(np.sum(image_feats ** 2, axis=1, keepdims=True))
    image_feats_ref = image_feats_ref / \
        np.sqrt(np.sum(image_feats_ref ** 2, axis=1, keepdims=True))
    res = image_feats @ image_feats_ref.T
    return np.mean(res)

parser = argparse.ArgumentParser()
parser.add_argument("--rank", type=int, default=0)
parser.add_argument("--world_size", type=int, default=1)
parser.add_argument("--result_root", type=str, default='', required=True)
parser.add_argument("--save_dir", type=str, default='', required=True)
args = parser.parse_args()

data_root = './datasets'
result_root = args.result_root
save_dir = os.path.join(args.save_dir, f'{os.path.basename(result_root)}')
os.makedirs(save_dir, exist_ok=True)

with open('./datasets/dreambench_singleip.json', 'r') as f:
    data = json.load(f)

device = 'cuda'
clip_model, _ = clip.load("ViT-B/32", device=device, jit=False)
clip_model.eval()
dino_model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16', pretrained=True).to(device)
dino_model.eval()
clipt_score, clipi_score, dino_score = [], [], []

comb_idx2prompt = {}
comb_idx=0
for idx, item in enumerate(data):
    if idx%25==0:
        comb_idx+=1
    item["result_path"] = f"{idx}"
    if comb_idx not in comb_idx2prompt:
        comb_idx2prompt[comb_idx] = []
    comb_idx2prompt[comb_idx].append(item)

clipt_score, clipi_score, dino_score = [], [], []
save_score_dict = {}
for i, (comb_idx, comb) in enumerate(comb_idx2prompt.items()):
    print(f'{i}/{len(comb_idx2prompt)}')
    print(comb_idx, comb)
    image_paths = []
    prompts = []
    entity1_concept_name = []
    entity2_concept_name = []
    for each in comb:
        img_path = each["result_path"]
        img_path = os.path.join(result_root, img_path+f'_{args.rank}.png')
        print(img_path)
        if os.path.exists(img_path):
            image_paths.append(img_path)
            prompts.append(each['text'])
            entity1_concept_name.append(os.path.join(data_root, each['image_path']))
    if not len(image_paths) > 0:
        print('skip')
        continue            
        
    comb_clipt_score, _ = \
        clipeval(image_paths, prompts, clip_model, device)
    entity1_concept_name = list(set(entity1_concept_name))
    assert len(entity1_concept_name)==1
    entity1_concept_name = '/'.join(entity1_concept_name[0].split('/')[:-1]) 
    comb_target_paths = [entity1_concept_name]

    comb_clipi_score, comb_dino_score = [], []    
    for i, comb_target_path in enumerate(comb_target_paths):
        cur_clipi_score = \
            clipeval_image(image_paths, comb_target_path, clip_model, device=device)
        cur_dino_score = \
            dinoeval_image(image_paths, comb_target_path, dino_model, device)
        comb_clipi_score.append(cur_clipi_score)
        comb_dino_score.append(cur_dino_score)
    comb_clipi_score = np.array(comb_clipi_score).mean()
    comb_dino_score = np.array(comb_dino_score).mean()

    clipt_score.append(comb_clipt_score)
    clipi_score.append(comb_clipi_score)
    dino_score.append(comb_dino_score)
    print(f'[rank: {args.rank}]', 'comb_idx : %d, clipt_score: %.4f, clipi_score: %.4f, dino_score: %.4f' % (comb_idx, comb_clipt_score, comb_clipi_score, comb_dino_score))
    save_score_dict[comb_idx]={
        'clipt_score': float(comb_clipt_score),
        'clipi_score': float(comb_clipi_score),
        'dino_score': float(comb_dino_score)
    }
    clipt_score_avg = np.array(clipt_score).mean()
    clipi_score_avg = np.array(clipi_score).mean()
    dino_score_avg = np.array(dino_score).mean()    
    print(f'[rank: {args.rank}]', f'avg: {clipt_score_avg}, {clipi_score_avg}, {dino_score_avg}')    
    save_score_dict[f'{comb_idx}_avg'] = {
        'clipt_score': float(clipt_score_avg),
        'clipi_score': float(clipi_score_avg),
        'dino_score': float(dino_score_avg),
        'comb': comb
    }    
    with open(os.path.join(save_dir, f'all_score_rank_{args.rank}.json'), 'w') as file:
        json.dump(save_score_dict, file, indent=4)