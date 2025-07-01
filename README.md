# t9 - Stable Diffusion Image Generator

A command-line tool for generating images using Stable Diffusion and LoRA models, with YAML configuration and flexible CLI options.

## Features
- Generate images with Stable Diffusion and LoRA models
- Supports YAML and command-line configuration
- Image2Image (img2img) support
- Saves images and metadata (YAML)

## Requirements
- Python 3.8+
- CUDA-enabled GPU (recommended for performance)
- Linux (tested)

## Installation
1. Clone this repository or copy the files to your workspace.
2. (Recommended) Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   **Or** install as a package (editable mode):
   ```bash
   pip install -e .
   ```

## Usage
Run the main script with desired arguments:
```bash
python main.py --model <model_path_or_id> --prompt "your prompt here" [other options]
```

Or use a YAML config:
```bash
python main.py --yaml config.yaml
```

### Common Arguments
- `--model`: Path or HuggingFace model id (e.g. `stabilityai/stable-diffusion-2-1`)
- `--prompt`: Text prompt for image generation
- `--lora`: Path to LoRA file (optionally with scale, e.g. `lora.safetensors:0.7`). Can be used multiple times.
- `--image`: Path to input image for img2img (optionally with strength, e.g. `input.png:0.6`)
- `--count`: Number of images to generate
- `--output`: Output directory
- `--width`, `--height`: Image size (default 512x512)
- `--num_inference_steps`: Denoising steps (default 20)
- `--guidance_scale`: CFG scale (default 7.5)

See `main.py` for all available arguments.

## Example
Generate 2 images with a LoRA model:
```bash
python main.py --model stabilityai/stable-diffusion-2-1 --prompt "a fantasy landscape" --lora mylora.safetensors:0.8 --count 2 --output results/
```

Image2Image example:
```bash
python main.py --model stabilityai/stable-diffusion-2-1 --prompt "a cyberpunk city" --image input.png:0.7 --output results/
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
python main.py --yaml config.yaml
```

## License
MIT
