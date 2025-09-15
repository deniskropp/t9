import argparse
import datetime
import os
import yaml
import logging
from typing import Optional
from PIL import Image
from core.generator import StableDiffusionGenerator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_config(yaml_path: Optional[str]) -> dict:
    """Loads configuration from a YAML file."""
    if not yaml_path or not os.path.exists(yaml_path):
        return {}
    try:
        with open(yaml_path, 'r') as f:
            return yaml.safe_load(f) or {}
    except yaml.YAMLError as e:
        logging.error(f"Error loading YAML config file {yaml_path}: {e}")
        return {}

def parse_and_merge_args() -> argparse.Namespace:
    """Parses command-line arguments and merges them with YAML configuration."""
    parser = argparse.ArgumentParser(description='Generate an image using Stable Diffusion and a LoRA model.')
    parser.add_argument('--count', type=int, default=1, help='Number of images to generate')
    parser.add_argument('--output', type=str, default='sd-output', help='Output path to save the images content and information')
    parser.add_argument('--model', type=str, default=None, help='Path to the base model')
    parser.add_argument('--prompt', type=str, default=None, help='The prompt for image generation')
    parser.add_argument('--lora', action='append', help='Path to the LoRA model (format: path:scale)')
    parser.add_argument('--lora_scale', type=float, default=None, help='Default weight applied with fused loras')
    parser.add_argument('--seed', type=int, default=None, help='Seed for random number generation')
    parser.add_argument('--width', type=int, default=None, help='Width of the output image')
    parser.add_argument('--height', type=int, default=None, help='Height of the output image')
    parser.add_argument('--num_inference_steps', type=int, default=None, help='Number of denoising steps')
    parser.add_argument('--guidance_scale', type=float, default=None, help='Guidance scale for classifier-free guidance')
    parser.add_argument('--yaml', type=str, default=None, help='Path to a YAML configuration file')
    parser.add_argument('--image', type=str, default=None, help='Path to the input image (format: path:strength)')

    args = parser.parse_args()
    yaml_config = load_config(args.yaml)

    # Merge logic: CLI args override YAML config
    # Set defaults only if not present in CLI args or YAML config
    defaults = {
        'model': 'stable-diffusion/v1-5',
        'prompt': 'a kinky hood in the forest',
        'lora': [],
        'lora_scale': 1.0,
        'seed': None,
        'width': 512,
        'height': 512,
        'num_inference_steps': 20,
        'guidance_scale': 7.5,
        'image': None,
        'output': 'sd-output'
    }

    # Apply defaults first
    for key, value in defaults.items():
        if getattr(args, key) is None and key not in yaml_config:
            setattr(args, key, value)

    # Apply YAML config, only if the attribute is still None (CLI arg not provided)
    for key, value in yaml_config.items():
        if getattr(args, key) is None:
            setattr(args, key, value)

    # Specific handling for lists and scales
    if args.lora is None: args.lora = []
    if yaml_config.get('lora'):
        args.lora.extend(yaml_config['lora'])

    # Ensure lora_scale is applied correctly if it was None initially
    if args.lora_scale is None:
        args.lora_scale = yaml_config.get('lora_scale', defaults['lora_scale'])

    # Ensure seed is handled correctly if it was None initially
    if args.seed is None:
        args.seed = yaml_config.get('seed', defaults['seed'])

    # Ensure image is handled correctly if it was None initially
    if args.image is None:
        args.image = yaml_config.get('image', defaults['image'])

    # Ensure output is handled correctly if it was None initially
    if args.output is None:
        args.output = yaml_config.get('output', defaults['output'])

    # Post-processing for image and lora arguments to parse scale/strength if needed
    # This logic is now handled within StableDiffusionGenerator setup

    logging.info(f"Final arguments after merging: {vars(args)}")
    return args


def setup_generator(args: argparse.Namespace) -> Optional[StableDiffusionGenerator]:
    """Sets up the StableDiffusionGenerator based on parsed arguments."""
    try:
        if not os.path.exists(args.output):
            os.makedirs(args.output)
            logging.info(f"Created output directory: {args.output}")

        model_path = args.model
        if not model_path: # Fallback if model arg is still None after merge
            logging.warning("Model path not specified, attempting default: './stable-diffusion/sd-v1-5.safetensors'")
            model_path = "./stable-diffusion/sd-v1-5.safetensors"
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Default model not found at {model_path}. Please specify --model.")

        generator = StableDiffusionGenerator(model_path=model_path)
        logging.info(f"Initialized StableDiffusionGenerator with model: {model_path}")

        if args.image:
            try:
                img_parts = args.image.split(':')
                image_filename = img_parts[0]
                image_strength = float(img_parts[1]) if len(img_parts) > 1 else 0.8
                init_image = Image.open(image_filename).convert("RGB")
                init_image = init_image.resize((args.width, args.height))
                generator.set_initial_image(image=init_image, strength=image_strength)
                logging.info(f"Set initial image: {image_filename} with strength {image_strength}")
            except FileNotFoundError:
                logging.error(f"Error: Input image file not found: {image_filename}")
                return None
            except Exception as e:
                logging.error(f"Error processing input image {image_filename}: {e}")
                return None

        lora_scale_default = args.lora_scale if args.lora_scale is not None else 1.0
        for lora_arg in args.lora:
            try:
                lora_parts = lora_arg.split(':')
                lora_filename = lora_parts[0]
                lora_scale = float(lora_parts[1]) if len(lora_parts) > 1 else lora_scale_default
                generator.load_lora_weights(lora_filename, lora_scale)
                logging.info(f"Loaded LoRA: {lora_filename} with scale {lora_scale}")
            except FileNotFoundError:
                logging.error(f"Error: LoRA file not found: {lora_filename}")
            except Exception as e:
                logging.error(f"Error loading LoRA {lora_filename}: {e}")

        generator.fuse(lora_scale=lora_scale_default)
        logging.info(f"Fused LoRA weights with scale: {lora_scale_default}")

        return generator

    except FileNotFoundError as e:
        logging.error(f"Setup failed: {e}")
        return None
    except Exception as e:
        logging.error(f"An unexpected error occurred during generator setup: {e}")
        return None

def generate_images(args: argparse.Namespace, generator: StableDiffusionGenerator):
    """Generates images using the configured StableDiffusionGenerator."""
    if not generator:
        logging.error("Cannot generate images, generator is not available.")
        return

    current_seed = args.seed
    for i in range(args.count):
        try:
            seed_to_use, timestamp = generator.prepare(current_seed)
            log_message = f"Generating image {i+1}/{args.count} | Seed: {seed_to_use} | Timestamp: {timestamp}"
            logging.info(log_message)
            print(f"\n----\n")
            print(f"Generation {i+1}/{args.count}:")
            print(f"  Timestamp: {datetime.datetime.now()}")
            print(f"  Seed: {seed_to_use}")
            print(f"  Size: {args.width}x{args.height}")
            print(f"  Model: {args.model}")
            print(f"  Prompt: {args.prompt}")
            print(f"  Image Input: {args.image}")
            print(f"  Steps: {args.num_inference_steps}")
            print(f"  Guidance Scale: {args.guidance_scale}")
            print(f"  LoRA models: {args.lora}")
            print(f"  LoRA scale: {args.lora_scale}")
            print(f"  Output dir: {args.output}")
            print(f"----\n")

            image = generator.run(
                seed=seed_to_use,
                timestamp=timestamp,
                output_dir=args.output,
                prompt=args.prompt,
                width=args.width,
                height=args.height,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale
            )

            # Increment seed for the next iteration if a specific seed was provided
            if current_seed is not None:
                current_seed += 1

        except Exception as e:
            logging.error(f"Error during image generation {i+1}/{args.count}: {e}")
            # Decide whether to continue or break based on error severity
            if args.count > 1: # If multiple images requested, try to continue
                logging.warning("Attempting to continue with the next image.")
            else: # If only one image, stop
                break


def main():
    """Main CLI entry point."""
    try:
        args = parse_and_merge_args()
        generator = setup_generator(args)
        generate_images(args, generator)
    except Exception as e:
        logging.critical(f"Critical error in main execution: {e}")

if __name__ == "__main__":
    main()
