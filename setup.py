from setuptools import setup, find_packages

setup(
    name="open-unlearning",
    version="0.1.0",
    author="Vineeth Dorna, Anmol Mekala",
    author_email="vineethdorna@gmail.com, m.anmolreddy@gmail.com",
    description="A library for machine unlearning in LLMs.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/locuslab/open-unlearning",
    license="MIT",
    packages=find_packages(),
    install_requires=[
        "huggingface-hub==0.29.1",
        "transformers==4.45.1",
        "numpy==2.2.3",
        "hydra-core==1.3",
        "hydra_colorlog==1.2.0",
        "torch==2.4.1",
        "datasets==3.0.1",
        "accelerate==0.34.2",
        "bitsandbytes==0.44.1",
        "rouge-score==0.1.2",
        "scipy==1.14.1",
        "tensorboard==2.18.0",
        "scikit-learn==1.5.2",
        "peft==0.17.1",
        "python-dotenv==1.2.1",
    ],
    extras_require={
        "lm-eval": [
            "lm-eval==0.4.8",
        ],  # Install using `pip install .[lm-eval]`
        "dev": [
            "pre-commit==4.0.1",
            "ruff==0.6.9",
        ],  # Install using `pip install .[dev]`
    },
    python_requires=">=3.11",
)
