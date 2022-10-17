# -*- coding: utf-8; mode: python -*-
import setuptools


with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pyhgl",
    version="0.1.0",
    author="jintaos2",
    author_email="jintaos2@zju.edu.cn",
    description="a Python Embedded Hardware Generation Language",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=None,
    packages=['pyhgl'],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: GNU General Public License (GPL)",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
) 

