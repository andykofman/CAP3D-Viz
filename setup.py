#!/usr/bin/env python3
"""
Setup script for CAP3D-Viz: High-Performance 3D Visualization for CAP3D Files
"""

from setuptools import setup, find_packages
import pathlib

# Read the README file
here = pathlib.Path(__file__).parent.resolve()
long_description = (here / "README.md").read_text(encoding="utf-8")

# Read version from the package
def get_version():
    with open("cap3d_viz/__init__.py", "r") as fp:
        for line in fp:
            if line.startswith("__version__"):
                return line.split('"')[1]
    raise RuntimeError("Unable to find version string.")

version = get_version()

setup(
    name="cap3d-viz",
    version=version,
    description="High-Performance 3D Visualization for CAP3D Files",
    long_description=long_description,
    long_description_content_type="text/markdown",
    
    # Author and project info
    author="Ahmed Ali",
    author_email="ali.a@aucegypt.edu",
    url="https://github.com/andykofman/RWCap_view",
    project_urls={
        "Bug Reports": "https://github.com/andykofman/RWCap_view/issues",
        "Source": "https://github.com/andykofman/RWCap_view",
        "Documentation": "https://github.com/andykofman/RWCap_view/docs",
    },
    
    # License and classification
    license="MIT",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Visualization",
        "Topic :: Scientific/Engineering :: Electronic Design Automation (EDA)",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
    
    # Keywords for discovery
    keywords=[
        "3D visualization", 
        "CAP3D", 
        "capacitance", 
        "integrated circuits", 
        "EDA", 
        "scientific computing",
        "plotly",
        "parsing",
        "state machine"
    ],
    
    # Package configuration
    packages=find_packages(),
    python_requires=">=3.8",
    
    # Dependencies
    install_requires=[
        "numpy>=1.19.0",
        "plotly>=5.0.0",
        "matplotlib>=3.3.0",
    ],
    
    # Optional dependencies
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov",
            "black",
            "flake8",
            "mypy",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme",
            "sphinx-autodoc-typehints",
        ],
        "performance": [
            "scipy>=1.7.0",
            "pandas>=1.3.0",
        ],
    },
    
    # Entry points for command-line usage
    entry_points={
        "console_scripts": [
            "cap3d-viz=cap3d_viz.cli:main",
        ],
    },
    
    # Include additional files
    include_package_data=True,
    package_data={
        "cap3d_viz": ["*.md", "*.txt"],
    },
    
    # Development status
    zip_safe=False,
)
