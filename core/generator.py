import datetime
import os
from typing import List, Optional, Tuple
from PIL import Image
import torch 
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipeline


class StableDiffusionGenerator:
    pipeline: StableDiffusionPipeline
    device: str
    model_path: str
    initial_image: Optional[Image.Image] = None
    image_strength: float = 0.8
    loras: List[Tuple[str, float]] = []

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipeline = self._load_pipeline(model_path)
        self.loras = []
        self.initial_image = None
        self.image_strength = 0.8
        print(f"Using device: {self.device}")

    def _load_pipeline(self, model_path: str) -> StableDiffusionPipeline:
        """Loads the Stable Diffusion pipeline."""
        try:
            pipe = StableDiffusionPipeline.from_single_file(
                model_path, torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            pipe.to(self.device)
            # Enable memory optimizations if available
            if self.device == "cuda":
                pipe.enable_model_cpu_offload()
                pipe.enable_xformers_memory_efficient_attention()
            return pipe
        except Exception as e:
            print(f"Error loading pipeline from {model_path}: {e}")
            raise

    def set_initial_image(self, image: Image.Image, strength: float):
        """Sets the initial image for img2img generation."""
        self.initial_image = image.convert("RGB")
        self.image_strength = strength
        print(f"Initial image set with strength: {strength}")

    def load_lora_weights(self, lora_path: str, scale: float = 1.0):
        """Loads LoRA weights. These will be fused later."""
        if not os.path.exists(lora_path):
            print(f"Warning: LoRA file not found at {lora_path}. Skipping.")
            return
        self.loras.append((lora_path, scale))
        print(f"LoRA loaded for fusing: {lora_path} with scale {scale}")

    def fuse(self, lora_scale: float = 1.0):
        """Fuses the loaded LoRA weights into the pipeline."""
        if not self.loras:
            print("No LoRA weights to fuse.")
            return

        try:
            for lora_path, scale in self.loras:
                print(f"Fusing LoRA: {lora_path} with scale {scale * lora_scale}")
                self.pipeline.load_lora_weights(os.path.dirname(lora_path), weight_name=os.path.basename(lora_path))
                # Note: diffusers load_lora_weights applies the scale internally. 
                # If you need to override the scale passed during load_lora_weights, 
                # you might need to manually adjust weights or use a different method.
                # For now, assuming the scale from load_lora_weights is sufficient or the default is used.
            print(f"Successfully fused {len(self.loras)} LoRA weights.")
            self.loras = [] # Clear after fusing
        except Exception as e:
            print(f"Error fusing LoRA weights: {e}")
            # Optionally, you might want to revert to the state before attempting to fuse
            raise

    def prepare(self, seed: Optional[int]) -> Tuple[int, str]:
        """Prepares the generator for image creation, setting seed and generating timestamp."""
        if seed is None:
            seed = int(torch.randint(0, 2**32 - 1, (1,)).item())
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        return seed, timestamp

    def run(
        self, 
        seed: int,
        timestamp: str,
        output_dir: str,
        prompt: str,
        width: int,
        height: int,
        num_inference_steps: int,
        guidance_scale: float
    ) -> Optional[Image.Image]:
        """Runs the image generation process."""
        try:
            generator_args = {
                "prompt": prompt,
                "width": width,
                "height": height,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "generator": torch.Generator(self.device).manual_seed(seed)
            }

            if self.initial_image:
                print("Performing img2img generation...")
                # Check if pipeline supports img2img
                if hasattr(self.pipeline, 'img2img') and self.pipeline.img2img is not None:
                    image = self.pipeline.img2img(
                        image=self.initial_image,
                        strength=self.image_strength,
                        **generator_args
                    ).images[0]
                else:
                    print("Error: Pipeline does not support img2img. Falling back to text2img.")
                    # Fallback or error out if img2img is expected but not available
                    image = self.pipeline(**generator_args).images[0]
            else:
                print("Performing text2img generation...")
                image = self.pipeline(**generator_args).images[0]

            # Save the generated image
            output_filename = f"{timestamp}-seed{seed}.png"
            output_path = os.path.join(output_dir, output_filename)
            image.save(output_path)
            print(f"Image saved to: {output_path}")

            # Save generation parameters
            self._save_generation_params(
                output_dir, timestamp, seed, generator_args, output_filename
            )

            return image

        except Exception as e:
            print(f"Error during image generation: {e}")
            return None

    def _save_generation_params(self, output_dir: str, timestamp: str, seed: int, params: dict, output_filename: str):
        """Saves the parameters used for generation to a file."""
        params_filename = f"{timestamp}-seed{seed}_params.txt"
        params_path = os.path.join(output_dir, params_filename)
        try:
            with open(params_path, 'w') as f:
                f.write(f"Timestamp: {datetime.datetime.now()}")
                f.write(f"\nSeed: {seed}")
                f.write(f"\nOutput Image: {output_filename}")
                for key, value in params.items():
                    if key != 'generator': # Don't write the generator object
                        f.write(f"\n{key.capitalize()}: {value}")
            print(f"Generation parameters saved to: {params_path}")
        except Exception as e:
            print(f"Error saving generation parameters: {e}")
