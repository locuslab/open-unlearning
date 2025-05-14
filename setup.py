from setuptools import setup, find_packages

# Read dependencies from requirements.txt
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

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
    install_requires=requirements,  # Uses requirements.txt
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
