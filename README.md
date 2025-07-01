# t9 - Stable Diffusion Image Generator

A flexible command-line tool for generating images using Stable Diffusion and LoRA models. Supports YAML configuration, advanced CLI options, and both text-to-image and image-to-image workflows.

## Features

- Generate images with Stable Diffusion and LoRA models
- YAML and command-line configuration
- Image2Image (img2img) support
- Saves images and metadata (YAML)

## Requirements

- Python 3.12+ (tested)
- CUDA-enabled GPU (recommended)

## Install

1. Clone this repository or copy the files to your workspace.
2. For `sd-make` (in your environment), install in editable mode:

```bash
pip install -e .
```

### Package

A non-editable install mode will be supported soon.

## Usage

Run the main script with desired arguments:

```bash
sd-make --model <model_path_or_id> --prompt "your prompt here" [other options]
```

Or use a YAML config:

```bash
sd-make --yaml config.yaml
```

### Common Arguments

- `--model`: Path or HuggingFace model id (e.g. `stabilityai/stable-diffusion-2-1`)
- `--prompt`: Text prompt for image generation
- `--lora`: Path to LoRA file (optionally with scale, e.g. `lora.safetensors:0.7`). Can be used multiple times.
- `--lora_scale`: Weight applied to fused LoRA models (default 1.0)
- `--image`: Path to input image for img2img (optionally with strength, e.g. `input.png:0.6`)
- `--count`: Number of images to generate (default 1)
- `--output`: Output directory (default `sd-output`)
- `--width`, `--height`: Image size in pixels (default 512x512)
- `--num_inference_steps`: Denoising steps (default 20)
- `--guidance_scale`: CFG scale (default 7.5)
- `--seed`: Seed for random number generation (optional)
- `--yaml`: Load parameters from YAML config file (optional)

See `main.py` for all available arguments and details.

## Examples

Generate 2 images with a LoRA model:

```bash
sd-make --model stabilityai/stable-diffusion-2-1 --prompt "a fantasy landscape" --lora mylora.safetensors:0.8 --count 2 --output results/
```

Image2Image example:

```bash
sd-make --model stabilityai/stable-diffusion-2-1 --prompt "a cyberpunk city" --image input.png:0.7 --output results/
```

YAML config example:

```yaml
model: stabilityai/stable-diffusion-2-1
prompt: "a futuristic robot"
lora:
  - mylora.safetensors:0.5
count: 1
output: results/
```

Run with:

```bash
sd-make --yaml config.yaml
```

## License

MIT
