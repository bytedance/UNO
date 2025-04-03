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
                    image_prompt1 = gr.Image(label="ref img1", visible=True, interactive=True, type="pil")
                    image_prompt2 = gr.Image(label="ref img2", visible=True, interactive=True, type="pil")
                    image_prompt3 = gr.Image(label="ref img3", visible=True, interactive=True, type="pil")
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

                with gr.Accordion("Generation Options", open=False):
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
    demo.launch(server_name="0.0.0.0", server_port=args.port)
