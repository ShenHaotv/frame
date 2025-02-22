#!/usr/bin/env python

from setuptools import setup

version = "1.0.0"

required = open("requirements.txt").read().split("\n")
with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="frame",
    version=version,
    description="Fine Resolution Asymmetric Migration Estimation (frame)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="[Hao Shen]",
    author_email="[shenhaotv@gmail.com]",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    url="https://github.com/shenhaotv/FRAME",
    packages=["frame"],
    install_requires=required,
    include_package_data=True,
    package_data={
        "": [
            "data/grid110.shp",
            "data/grid110.shx",
            "data/grid220.shp",
            "data/grid220.shx",
            "data/grid440.shp",
            "data/grid440.shx",
            "data/grid880.shp",
            "data/grid880.shx",
            "data/wolf.bed",
            "data/wolf.bim",
            "data/wolf.fam",
            "data/wolf.coord",
            "data/wolf.outer",
        ]
    },
    license="MIT",
)
