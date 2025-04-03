<h3 align="center">
    <img src="assets/logo.png" alt="Logo" style="vertical-align: middle; width: 40px; height: 40px;">
    Less-to-More Generalization: Unlocking More Controllability by In-Context Generation
</h3>

<p align="center"> 
<a href="https://bytedance.github.io/UNO//"><img alt="Build" src="https://img.shields.io/badge/Project%20Page-UNO-yellow"></a> 
<a href="https://huggingface.co/bytedance-research/UNO"><img src="https://img.shields.io/static/v1?label=%F0%9F%A4%97%20Hugging%20Face&message=Model&color=orange"></a>
</p>

><p align="center"> <span style="color:#137cf3; font-family: Gill Sans">Shaojin Wu,</span><sup></sup></a>  <span style="color:#137cf3; font-family: Gill Sans">Mengqi Huang</span><sup>*</sup>,</a> <span style="color:#137cf3; font-family: Gill Sans">Wenxu Wu,</span><sup></sup></a>  <span style="color:#137cf3; font-family: Gill Sans">Yufeng Cheng,</span><sup></sup> </a>  <span style="color:#137cf3; font-family: Gill Sans">Fei Ding</span><sup>+</sup>,</a> <span style="color:#137cf3; font-family: Gill Sans">Qian He</span></a> <br> 
><span style="font-size: 16px">Intelligent Creation Team, ByteDance</span></p>

<p align="center">
<img src="./assets/teaser.jpg" width=95% height=95% 
class="center">
</p>

## 🔥 News
- [04/2025] 🔥 The inference code is available in this repo. The [training code](https://github.com/bytedance/UNO), [model](https://huggingface.co/bytedance-research/UNO), and [demo](https://huggingface.co/spaces/bytedance-research/UNO-FLUX) will coming soon.
- [04/2025] 🔥 The [project page](https://bytedance.github.io/UNO) of UNO is created. The paper of UNO will be released on arXiv.

## 📖 Introduction
In this study, we propose a highly-consistent data synthesis pipeline to tackle this challenge. This pipeline harnesses the intrinsic in-context generation capabilities of diffusion transformers and generates high-consistency multi-subject paired data. Additionally, we introduce UNO, which consists of progressive cross-modal alignment and universal rotary position embedding. It is a multi-image conditioned subject-to-image model iteratively trained from a text-to-image model. Extensive experiments show that our method can achieve high consistency while ensuring controllability in both single-subject and multi-subject driven generation.


## ⚡️ Quick Start

## 🔧 Requirements and Installation

Install the requirements
```bash
## create a virtual environment with python >= 3.10 <= 3.12, like
# python -m venv uno_env
# source uno_env/bin/activate
# then install
pip install -r requirements.txt
```

then download checkpoints in one of the three ways:
1. Directly run the inference scripts, the checkpoints will be downloaded automatically by the `hf_hub_download` function in the code to your `$HF_HOME`(the default value is `~/.cache/huggingface`).
2. use `huggingface-cli download <repo name>` to download `black-forest-labs/FLUX.1-dev`, `xlabs-ai/xflux_text_encoders`, `openai/clip-vit-large-patch14`, `TODO UNO hf model`, then run the inference scripts.
3. use `huggingface-cli download <repo name> --local-dir <LOCAL_DIR>` to download all the checkpoints menthioned in 2. to the directories your want. Then set the environment variable `TODO`. Finally, run the inference scripts.

### 🌟 Gradio Demo

```bash
python app.py
```


### ✍️ Inference

- Optional prepreration: If you want to test the inference on dreambench at the first time, you should clone the submodule `dreambench` to download the dataset.

```bash
git submodule update --init
```


```bash
python inference.py
```

### 🚄 Training

```bash
accelerate launch train.py
```


## 🔥Updates
We will open source this project as soon as possible. Thank you for your patience and support! 🌟
- [x] Release github repo.
- [x] Release inference code.
- [ ] Release model checkpoints.
- [ ] Release arXiv paper.
- [ ] Release training code.
- [ ] Release in-context data generation pipelines.

##  Citation
If UNO is helpful, please help to ⭐ the repo.
