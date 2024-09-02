#!/usr/bin/env python

from setuptools import setup

version = "1.0.0"

required = open("requirements.txt").read().split("\n")
with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="FRAME",
    version=version,
    description="Fine Resolution Asymmetric Migration Estimation (FRAME)",
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
    packages=["FRAME"],
    install_requires=required,
    include_package_data=True,
    package_data={
        "": [
            "data/grid100.shp",
            "data/grid100.shx",
            "data/grid250.shp",
            "data/grid250.shx",
            "data/grid500.shp",
            "data/grid500.shx",
            "data/wolvesadmix.bed",
            "data/wolvesadmix.coord",
            "data/wolvesadmix.fam",
            "data/wolvesadmix.bim",
            "data/wolvesadmix.outer",
            "data/warbler.csv"
            "data/warbler.coord"
            "data/warbler.outer"
        ]
    },
    license="MIT",
)