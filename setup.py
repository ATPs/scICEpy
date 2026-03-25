from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="scICEpy",
    version="0.1.0",
    author="Xiaolong Cao",
    author_email="atps@outlook.com",
    description="Single-cell Inconsistency-based Clustering Evaluation for Python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ATPs/scICEpy",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "scanpy>=1.9.0",
        "anndata>=0.8.0",
        "scipy>=1.7.0",
        "matplotlib>=3.4.0",
        "python-igraph>=0.10.0",
        "leidenalg>=0.9.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
        ],
    },
)
