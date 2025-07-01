import datetime
import os
import yaml
from PIL import Image
import torch
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline


class StableDiffusionGenerator:
    device: str
    model: str
    init_image: Image.Image
    strength: float
    image_filename: str
    loras: list[str]
    adapter_names: list[str]
    adapter_weights: list[float]
    generator: torch.Generator
    pipe: StableDiffusionPipeline

    def __init__(self, model_id):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using {self.device} device")

        num_threads = int(os.environ.get('NUM_THREADS', -1))
        if num_threads < 0:
            num_threads = os.cpu_count() or 1
        print(f"Using {num_threads} threads")
        if num_threads > 1:
            torch.set_num_threads(num_threads)
            torch.set_num_interop_threads(num_threads)
        else:
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)

        self.model = model_id
        self.init_image = None
        self.strength = 0.8
        self.image_filename = None
        self.adapter_names = []
        self.adapter_weights = []
        self.loras = []
        self.reset_pipe()

    def reset_pipe(self):
        self.pipe = None

    def init_pipe(self):
        if (self.pipe is not None):
            return

        print(f"Using {self.device} device")

        torch_dtype = torch.bfloat16 if self.device == "cuda" else torch.float32

        if (os.path.isfile(self.model)):
            if (self.init_image):
                self.pipe = StableDiffusionImg2ImgPipeline.from_single_file(
                    self.model,
                    torch_dtype=torch_dtype,
                    safety_checker=None
                )
            else:
                self.pipe = StableDiffusionPipeline.from_single_file(
                    self.model,
                    torch_dtype=torch_dtype,
                    safety_checker=None
                )
        else:
            if (self.init_image):
                self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                    self.model,
                    torch_dtype=torch_dtype,
                    safety_checker=None
                )
            else:
                self.pipe = StableDiffusionPipeline.from_pretrained(
                    self.model,
                    torch_dtype=torch_dtype,
                    safety_checker=None
                )

        self.pipe = self.pipe.to(self.device)
        self.pipe.safety_checker = None

    def set_initial_image(self, image, strength=0.8, filename=None):
        self.init_image = image
        self.strength = strength
        self.image_filename = filename
        self.reset_pipe()

    def load_lora_weights(self, lora, scale=1.0):
        self.init_pipe()
        self.loras.append(f"{lora}:{scale}")
        self.pipe.load_lora_weights(".", weight_name=lora)
        for l in self.pipe.get_active_adapters():
            self.adapter_names.append(l)
            self.adapter_weights.append(scale)

    def fuse(self, lora_scale=0.5):
        self.init_pipe()
        self.lora_scale = lora_scale
        if self.adapter_names:
            self.pipe.set_adapters(self.adapter_names, adapter_weights=self.adapter_weights)
            self.pipe.fuse_lora(lora_scale=lora_scale, adapter_names=self.adapter_names)
            self.pipe.unload_lora_weights()
        self.generator = torch.Generator(device=self.device)

    def generate_image(self, prompt, width, height, num_inference_steps, guidance_scale) -> Image:
        self.init_pipe()
        return self.pipe(
            prompt,
            image=self.init_image,
            strength=self.strength,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=self.generator
        ).images[0]

    def save_image_and_info(self, image, output_path_noext, timestamp, seed,
                            prompt, width, height, num_inference_steps, guidance_scale):
        image.save(f"{output_path_noext}_{timestamp}.png")
        image_info = {
            'model': self.model,
            'lora': self.loras,
            'lora_scale': self.lora_scale,
            'image': f"{self.image_filename}:{self.strength}" if self.image_filename else None,
            'prompt': prompt,
            'width': width,
            'height': height,
            'num_inference_steps': num_inference_steps,
            'guidance_scale': guidance_scale,
            'timestamp': timestamp,
            'seed': seed,
        }
        with open(f"{output_path_noext}_{timestamp}.yaml", 'w') as file:
            yaml.dump(image_info, file)

    def prepare(self, seed):
        if seed is None:
            seed = self.generator.seed()
        else:
            self.generator = self.generator.manual_seed(seed)
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        return seed, timestamp

    def run(self, seed, timestamp, output, prompt, width, height, num_inference_steps, guidance_scale):
        image = self.generate_image(prompt, width, height, num_inference_steps, guidance_scale)
        self.save_image_and_info(image, f"{output}/sd_", timestamp, seed,
                                 prompt, width, height, num_inference_steps, guidance_scale)
        return image
