from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="adaptive-inference-batching",
    version="0.1.0",
    description="Adaptive Inference Batching using Policy Gradients",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/surlac/adaptive-inference-batching",
    packages=find_packages(),
    package_data={
        "": ["*.yaml"],  # Include any YAML config files
    },
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: MacOS",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    install_requires=[
        "numpy>=1.21.0",
        "torch>=2.0.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "tqdm>=4.62.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.2.0",
        "pyyaml>=6.0",
        "gym>=0.26.0",
    ],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "train-rl=scripts.run_training:main",
            "evaluate-rl=scripts.evaluation.evaluate_enhanced:main",
            "generate-report=scripts.generate_report_data:main",
        ],
    },
    extras_require={
        "dev": [
            "pytest>=7.3.0",
            "pytest-cov>=4.0.0",
            "black>=23.3.0",
            "flake8>=6.0.0",
            "jupyter>=1.0.0",
            "ipykernel>=6.0.0",
        ],
    },
    include_package_data=True,
)
