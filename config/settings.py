import yaml
from typing import Optional
from argparse import Namespace

class Settings:
    def __init__(self, args: Namespace, yaml_args: Namespace):
        self.args = args
        self.yaml_args = yaml_args
        self.settings = self._merge_settings()

    def _merge_settings(self) -> Namespace:
        merged = self.args

        # Apply YAML arguments, ensuring lists are handled correctly
        if self.yaml_args:
            for key, value in vars(self.yaml_args).items():
                if value is not None:
                    if key == 'lora' and isinstance(value, list):
                        # If args.lora is empty or None, use yaml_args.lora
                        # If args.lora has items, extend it with yaml_args.lora
                        if not getattr(merged, 'lora', None):
                            setattr(merged, 'lora', [])
                        merged.lora.extend(value)
                    elif getattr(merged, key, None) is None or getattr(merged, key) == parser.get_default(key):
                        setattr(merged, key, value)

        # Apply defaults and ensure essential parameters are set
        merged.model = merged.model or 'stable-diffusion/v1-5'
        merged.prompt = merged.prompt or 'a kinky hood in the forest'
        merged.lora = merged.lora or []
        merged.lora_scale = merged.lora_scale if merged.lora_scale is not None else 1.0
        merged.seed = merged.seed if merged.seed is not None else None
        merged.width = merged.width if merged.width is not None else 512
        merged.height = merged.height if merged.height is not None else 512
        merged.num_inference_steps = merged.num_inference_steps if merged.num_inference_steps is not None else 20
        merged.guidance_scale = merged.guidance_scale if merged.guidance_scale is not None else 7.5
        merged.image = merged.image
        merged.output = merged.output or 'sd-output'

        # Remove duplicates from lora list if any
        if hasattr(merged, 'lora'):
            merged.lora = list(dict.fromkeys(merged.lora))

        return merged

    def get_settings(self) -> Namespace:
        return self.settings

def load_config(args):
    yaml_args = Namespace()
    if args.yaml:
        try:
            with open(args.yaml, 'r') as f:
                yaml_data = yaml.safe_load(f)
            if yaml_data:
                yaml_args.__dict__.update(yaml_data)
        except FileNotFoundError:
            print(f"Warning: YAML config file not found at {args.yaml}")
        except yaml.YAMLError as e:
            print(f"Warning: Error parsing YAML config file: {e}")

    return Settings(args, yaml_args)
