from setuptools import setup, find_packages

setup(
    name="t9-stable-diffusion",
    version="0.1.0",
    py_modules=["main"],
    packages=find_packages(),
    install_requires=[
        "diffusers",
        "torch",
        "Pillow",
        "pyyaml",
    ],
    entry_points={
        "console_scripts": [
            "sd-make=main:main",
        ],
    },
)