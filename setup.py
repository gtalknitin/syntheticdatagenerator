"""Setup configuration for Synthetic Data Generator package."""

from setuptools import setup, find_packages
from pathlib import Path

# Read the contents of README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

# Read requirements
requirements = []
with open("requirements.txt") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

dev_requirements = []
with open("requirements-dev.txt") as f:
    dev_requirements = [
        line.strip()
        for line in f
        if line.strip() and not line.startswith("#") and not line.startswith("-r")
    ]

setup(
    name="synthetic-data-generator",
    version="9.0.0",
    author="NikAlgoBulls",
    author_email="",
    description="High-quality synthetic options market data generator for algorithmic trading backtests",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    project_urls={
        "Documentation": "",
        "Source": "",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": dev_requirements,
    },
    entry_points={
        "console_scripts": [
            "synth-data=synthetic_data_generator.cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
