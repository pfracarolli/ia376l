from setuptools import find_packages, setup

setup(
    name="ia376l",
    packages=find_packages(),
    version="0.1.0",
    description="Final project for Unicamp's IA376-L Deep Learning for Signal Synthesis course.",
    author="pfracarolli",
    license="",
    install_requires=["pytorch_lightning", "einops"],
)
