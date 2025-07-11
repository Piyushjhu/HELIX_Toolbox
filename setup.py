#!/usr/bin/env python3
"""
Setup script for HELIX Toolbox
A Comprehensive GUI for Single Point PDV Data Analysis
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="helix-toolbox",
    version="1.0.0",
    author="Piyush Wanchoo",
    author_email="pwanchoo@jhu.edu",
    description="A Comprehensive GUI for Single Point PDV Data Analysis",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/Piyushjhu/HELIX_Toolbox",
    project_urls={
        "Bug Reports": "https://github.com/Piyushjhu/HELIX_Toolbox/issues",
        "Source": "https://github.com/Piyushjhu/HELIX_Toolbox",
        "Documentation": "https://github.com/Piyushjhu/HELIX_Toolbox#readme",
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Visualization",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
        "Environment :: X11 Applications :: GTK",
        "Environment :: X11 Applications :: Qt",
    ],
    python_requires=">=3.7",
    packages=find_packages(),
    include_package_data=True,
    install_requires=read_requirements(),
    entry_points={
        "console_scripts": [
            "helix-toolbox=alpss_spade_gui:main",
        ],
    },
    keywords=[
        "pdv", "photonic doppler velocimetry", "spall", "shock physics",
        "velocity interferometry", "signal processing", "gui", "analysis"
    ],
    platforms=["any"],
    license="MIT",
    zip_safe=False,
) 