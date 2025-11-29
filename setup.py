"""Setup script for backward compatibility with older tools."""

from setuptools import setup, find_packages
import os

# Read version from _version.py
version = {}
with open(os.path.join("pyehicle", "_version.py")) as f:
    exec(f.read(), version)

# Read README
with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="pyehicle",
    version=version["__version__"],
    author="Nick Doulos",
    author_email="contact@nickdoulos.com",
    description="A Python library for GPS trajectory processing and reconstruction using road network data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Nick-Doulos/pyehicle",
    project_urls={
        "Bug Tracker": "https://github.com/Nick-Doulos/pyehicle/issues",
        "Documentation": "https://github.com/Nick-Doulos/pyehicle#readme",
        "Source Code": "https://github.com/Nick-Doulos/pyehicle",
    },
    packages=find_packages(exclude=["tests", "tests.*", "docs", "examples"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Programming Language :: Python :: 3.14",
        "Topic :: Scientific/Engineering :: GIS",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "pyproj>=3.0.0",
        "shapely>=1.8.0",
        "igraph>=0.9.0",
        "rtree>=0.9.0",
        "scikit-learn>=1.0.0",
        "osmium>=3.0.0",
        "folium>=0.12.0",
        "matplotlib>=3.3.0",
        "branca>=0.4.0",
        "requests>=2.25.0",
        "geopandas>=0.10.0",
        "pyrosm>=0.6.0",
        "polars>=0.15.0",
        "numba>=0.55.0",
        "leuvenmapmatching>=1.0.0",
        "osmnx>=1.1.0",
        "tqdm>=4.60.0",
    ],
    keywords="gps trajectory map-matching road-network trajectory-reconstruction geospatial location-data",
    include_package_data=True,
    zip_safe=False,
)
