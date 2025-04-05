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

import gradio as gr
import torch

from uno.flux.pipeline import UNOPipeline


def create_demo(
    model_type: str,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    offload: bool = False,
):
    pipeline = UNOPipeline(model_type, device, offload, only_lora=True, lora_rank=512)

    with gr.Blocks() as demo:
        gr.Markdown(f"# UNO by UNO team")
        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(label="Prompt", value="handsome woman in the city")
                with gr.Row():
                    image_prompt1 = gr.Image(label="Ref Img1", visible=True, interactive=True, type="pil")
                    image_prompt2 = gr.Image(label="Ref Img2", visible=True, interactive=True, type="pil")
                    image_prompt3 = gr.Image(label="Ref Img3", visible=True, interactive=True, type="pil")
                    image_prompt4 = gr.Image(label="ref img4", visible=True, interactive=True, type="pil")

                with gr.Row():
                    with gr.Column():
                        ref_long_side = gr.Slider(128, 512, 512, step=16, label="Long side of Ref Images")
                    with gr.Column():
                        gr.Markdown("📌 **The recommended ref scale** is related to the ref img number.\n")
                        gr.Markdown("   1->512 / 2->320 / 3...n->256")

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
                seed, ref_long_side, image_prompt1, image_prompt2, image_prompt3, image_prompt4
            ]
            generate_btn.click(
                fn=pipeline.gradio_generate,
                inputs=inputs,
                outputs=[output_image, download_btn],
            )
        
        example_text = gr.Text("", visible=False, label="Case For:")

        example_three_ip = [
            "Many2One",
            "A woman wears the dress and holds a bag, in the flowers",
            "assets/examples/many2one/ref1.png",
            "assets/examples/many2one/ref2.png",
            "assets/examples/many2one/ref3.png",
            72,
            256,
            "assets/examples/many2one/result.png",
        ]

        gr.Examples(
            examples=[
                example_three_ip,
            ],
            inputs=[
                example_text, prompt, image_prompt1, image_prompt2, image_prompt3, seed, ref_long_side, output_image
            ]
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
