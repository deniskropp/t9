import argparse

def setup_argument_parser():
    parser = argparse.ArgumentParser(description='Generate an image using Stable Diffusion and a LoRA model.')
    parser.add_argument('--count', type=int, default=1, help='Number of images to generate')
    parser.add_argument('--output', type=str, default='sd-output', help='Output path to save the images content and information')
    parser.add_argument('--model', type=str, default=None, help='Path to the base model')
    parser.add_argument('--prompt', type=str, default=None, help='The prompt for image generation')
    parser.add_argument('--lora', action='append', help='Path to the LoRA model')
    parser.add_argument('--lora_scale', type=float, default=None, help='Weight applied with fused loras at the output')
    parser.add_argument('--seed', type=int, default=None, help='Seed for random number generation')
    parser.add_argument('--width', type=int, default=None, help='Width of the output image')
    parser.add_argument('--height', type=int, default=None, help='Height of the output image')
    parser.add_argument('--num_inference_steps', type=int, default=None, help='Number of denoising steps')
    parser.add_argument('--guidance_scale', type=float, default=None, help='Guidance scale for classifier-free guidance')
    parser.add_argument('--yaml', type=str, default=None, help='Load parameters from YAML (args or sd_)')
    parser.add_argument('--image', type=str, default=None, help='Path to the input image')
    return parser

def parse_arguments(parser):
    return parser.parse_args()
