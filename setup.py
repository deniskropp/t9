from setuptools import setup, find_packages

setup(
    name="t9-stable-diffusion",
    version="0.1.0",
    py_modules=["main"],
    install_requires=[
        "diffusers",
        "torch",
        "Pillow",
        "pyyaml",
        # add any other dependencies from requirements.txt
    ],
    entry_points={
        "console_scripts": [
            "sd-make=main:main",
        ],
    },
)