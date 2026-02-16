"""
Setup script for Spatiotemporal AV Navigator
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="spatiotemporal-av-navigator",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Geographically weighted learning for autonomous vehicle navigation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/Spatiotemporal-AV-Navigator",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: GIS",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "gwlearn>=0.1.0",
        "geopandas>=0.14.0",
        "shapely>=2.0.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "pandas>=2.0.0",
        "libpysal>=4.9.0",
        "osmnx>=1.9.0",
        "networkx>=3.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "black>=23.0",
            "flake8>=6.0",
            "mypy>=1.0",
        ],
        "visualization": [
            "plotly>=5.14.0",
            "folium>=0.14.0",
        ],
        "simulation": [
            "pygame>=2.5.0",
            "imageio>=2.31.0",
            "opencv-python>=4.8.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "gw-navigator=src.gw_navigator:main",
        ],
    },
)
