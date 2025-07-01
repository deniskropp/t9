import argparse
import csv
import datetime
import json
import os
import yaml

from PIL import Image
from sd.generator import StableDiffusionGenerator


def parse_args():
    """Parse command-line arguments and return argparse.Namespace."""
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


def merge_args_with_yaml(args, parser):
    """Merge command-line args with YAML file if provided. CLI args take precedence."""
    # Parse initial args
    yaml_args = parser.parse_args()
    if yaml_args.lora is None:
        yaml_args.lora = []
    # Load YAML if provided
    if yaml_args.yaml:
        with open(yaml_args.yaml, 'r') as f:
            yaml_data = yaml.safe_load(f)
        if yaml_data:
            yaml_args.__dict__.update(yaml_data)
    # Parse again to get CLI overrides
    args = parser.parse_args()
    # Set defaults and merge
    args.model = args.model or yaml_args.model or 'stable-diffusion/v1-5'
    args.prompt = args.prompt or yaml_args.prompt or 'a kinky hood in the forest'
    args.lora = args.lora or []
    if yaml_args.lora:
        args.lora.extend(yaml_args.lora)
    args.lora_scale = args.lora_scale if args.lora_scale is not None else (yaml_args.lora_scale if yaml_args.lora_scale else 1.0)
    args.seed = args.seed if args.seed is not None else (yaml_args.seed if yaml_args.seed else None)
    args.width = args.width if args.width is not None else (yaml_args.width if yaml_args.width else 512)
    args.height = args.height if args.height is not None else (yaml_args.height if yaml_args.height else 512)
    args.num_inference_steps = args.num_inference_steps if args.num_inference_steps is not None else (yaml_args.num_inference_steps if yaml_args.num_inference_steps else 20)
    args.guidance_scale = args.guidance_scale if args.guidance_scale is not None else (yaml_args.guidance_scale if yaml_args.guidance_scale else 7.5)
    args.image = args.image or yaml_args.image
    args.output = args.output or yaml_args.output or 'sd-output'
    return args, yaml_args


def setup_generator(args, yaml_args):
    """Initialize StableDiffusionGenerator and set up image and LoRA weights."""
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    model_path = args.model if args.model else "./stable-diffusion/sd-v1-5.safetensors"
    generator = StableDiffusionGenerator(model_path)
    # Set initial image if provided
    image_arg = args.image or yaml_args.image
    if image_arg:
        image_info = image_arg.split(':')
        image_filename = image_info[0]
        image_strength = float(image_info[1]) if len(image_info) > 1 else 0.8
        init_image = Image.open(image_filename)
        if init_image is None:
            print(f"Error: Failed to open image file {image_filename}")
        else:
            init_image = init_image.resize((args.width, args.height))
            generator.set_initial_image(image=init_image, strength=image_strength, filename=image_filename)
    # Load LoRA weights
    for lora in args.lora:
        lora_info = lora.split(':')
        lora_filename = lora_info[0]
        lora_scale = float(lora_info[1]) if len(lora_info) > 1 else 1.0
        generator.load_lora_weights(lora_filename, lora_scale)
    generator.fuse(lora_scale=args.lora_scale if args.lora_scale is not None else 1.0)
    return generator


def generate_images(args, generator):
    """Generate images and print info for each."""
    for i in range(args.count):
        seed, timestamp = generator.prepare(args.seed)
        print('\n----\n')
        print(f"Date and time: {datetime.datetime.now()}")
        print(f"Seed: {args.seed}")
        print(f"Size: {args.width}x{args.height}")
        print(f"Model: {args.model}")
        print(f"Prompt: {args.prompt}")
        print(f"Image: {args.image}")
        print(f"Steps: {args.num_inference_steps}")
        print(f"Guidance Scale: {args.guidance_scale}")
        print(f"LoRA models: {args.lora}")
        print(f"LoRA scale: {args.lora_scale}")
        print(f"Seed: {seed}")
        print(f"Timestamp: {timestamp}")
        print(f"Output: {args.output}")
        print(f"---- {i+1}/{args.count}\n")
        image = generator.run(seed, timestamp, args.output, args.prompt, args.width, args.height, args.num_inference_steps, args.guidance_scale)
        if args.seed is not None:
            args.seed += 1


def run_cli():
    """Main CLI entry point."""
    parser = parse_args()
    args, yaml_args = merge_args_with_yaml(None, parser)
    generator = setup_generator(args, yaml_args)
    generate_images(args, generator)


if __name__ == "__main__":
    run_cli()
