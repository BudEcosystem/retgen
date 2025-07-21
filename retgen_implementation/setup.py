from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="retgen",
    version="0.1.0",
    author="RETGEN Implementation",
    description="Retrieval-Enhanced Text Generation through Vector Database Emulation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/retgen",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "torch>=2.0.0",
        "sentence-transformers>=2.2.0",
        "faiss-cpu>=1.7.3",
        "transformers>=4.30.0",
        "datasets>=2.14.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.3.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
        "gpu": ["faiss-gpu>=1.7.3"],
    },
    entry_points={
        "console_scripts": [
            "retgen-train=retgen.scripts.train:main",
            "retgen-generate=retgen.scripts.generate:main",
            "retgen-benchmark=retgen.scripts.benchmark:main",
        ],
    },
)