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

import dataclasses
import json
from pathlib import Path

import gradio as gr
import torch

from uno.flux.pipeline import UNOPipeline


def get_examples(examples_dir: str = "assets/examples") -> list:
    examples = Path(examples_dir)
    ans = []
    for example in examples.iterdir():
        if not example.is_dir():
            continue
        with open(example / "config.json") as f:
            example_dict = json.load(f)
  
        
        example_list = []

        example_list.append(example_dict["useage"])  # case for
        example_list.append(example_dict["prompt"])  # prompt

        for key in ["image_ref1", "image_ref2", "image_ref3", "image_ref4"]:
            if key in example_dict:
                example_list.append(str(example / example_dict[key]))
            else:
                example_list.append(None)

        example_list.append(example_dict["seed"])

        ans.append(example_list)
    return ans


def create_demo(
    model_type: str,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    offload: bool = False,
):
    pipeline = UNOPipeline(model_type, device, offload, only_lora=True, lora_rank=512)

    badges_text = r"""
    <div style="text-align: center; display: flex; justify-content: left; gap: 5px;">
    <a href="https://bytedance.github.io/UNO/"><img alt="Build" src="https://img.shields.io/badge/Project%20Page-UNO-yellow"></a> 
    <a href="https://arxiv.org/abs/2504.02160"><img alt="Build" src="https://img.shields.io/badge/arXiv%20paper-UNO-b31b1b.svg"></a>
    <a href="https://huggingface.co/bytedance-research/UNO"><img src="https://img.shields.io/static/v1?label=%F0%9F%A4%97%20Hugging%20Face&message=Model&color=orange"></a>
    <a href="https://huggingface.co/spaces/bytedance-research/UNO-FLUX"><img src="https://img.shields.io/static/v1?label=%F0%9F%A4%97%20Hugging%20Face&message=demo&color=orange"></a>
    </div>
    """.strip()

    with gr.Blocks() as demo:
        gr.Markdown(f"# UNO by UNO team")
        gr.Markdown(badges_text)
        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(label="Prompt", value="handsome woman in the city")
                with gr.Row():
                    image_prompt1 = gr.Image(label="Ref Img1", visible=True, interactive=True, type="pil")
                    image_prompt2 = gr.Image(label="Ref Img2", visible=True, interactive=True, type="pil")
                    image_prompt3 = gr.Image(label="Ref Img3", visible=True, interactive=True, type="pil")
                    image_prompt4 = gr.Image(label="Ref img4", visible=True, interactive=True, type="pil")

                with gr.Row():
                    with gr.Column():
                        width = gr.Slider(512, 2048, 512, step=16, label="Gneration Width")
                        height = gr.Slider(512, 2048, 512, step=16, label="Gneration Height")
                    with gr.Column():
                        gr.Markdown("📌 The model trained on 512x512 resolution.\n")
                        gr.Markdown(
                            "The size closer to 512 is more stable,"
                            " and the higher size gives a better visual effect but is less stable"
                        )

                with gr.Accordion("Advanced Options", open=False):
                    with gr.Row():
                        num_steps = gr.Slider(1, 50, 25, step=1, label="Number of steps")
                        guidance = gr.Slider(1.0, 5.0, 4.0, step=0.1, label="Guidance", interactive=True)
                        seed = gr.Number(-1, label="Seed (-1 for random)")

                generate_btn = gr.Button("Generate")

            with gr.Column():
                output_image = gr.Image(label="Generated Image")
                download_btn = gr.File(label="Download full-resolution", type="filepath", interactive=False)


            inputs = [
                prompt, width, height, guidance, num_steps,
                seed, image_prompt1, image_prompt2, image_prompt3, image_prompt4
            ]
            generate_btn.click(
                fn=pipeline.gradio_generate,
                inputs=inputs,
                outputs=[output_image, download_btn],
            )
        
        example_text = gr.Text("", visible=False, label="Case For:")
        examples = get_examples("./assets/examples")

        gr.Examples(
            examples=examples,
            inputs=[
                example_text, prompt,
                image_prompt1, image_prompt2, image_prompt3, image_prompt4,
                seed, output_image
            ],
        )

    return demo

if __name__ == "__main__":
    from typing import Literal

    from transformers import HfArgumentParser

    @dataclasses.dataclass
    class AppArgs:
        name: Literal["flux-dev", "flux-dev-fp8", "flux-schnell"] = "flux-dev"
        device: Literal["cuda", "cpu"] = "cuda" if torch.cuda.is_available() else "cpu"
        offload: bool = dataclasses.field(
            default=False,
            metadata={"help": "If True, sequantial offload the models(ae, dit, text encoder) to CPU if not used."}
        )
        port: int = 7860

    parser = HfArgumentParser([AppArgs])
    args_tuple = parser.parse_args_into_dataclasses() # type: tuple[AppArgs]
    args = args_tuple[0]

    demo = create_demo(args.name, args.device, args.offload)
    demo.launch(server_port=args.port)
