# Lint as: python3
""" HuggingFace/AutoNLP
"""

import datetime
import itertools
import os
import sys

from setuptools import find_packages
from setuptools import setup

DOCLINES = __doc__.split("\n")

setup(
    name="autonlp",
    version="0.0.1",
    description=DOCLINES[0],
    long_description="\n".join(DOCLINES[2:]),
    author="HuggingFace Inc.",
    author_email="abhishek@huggingface.co",
    url="https://github.com/huggingface/autonlp",
    download_url="https://github.com/huggingface/autonlp/tags",
    license="Apache 2.0",
    package_dir={"": "src"},
    packages=find_packages("src"),
    scripts=["scripts/autonlp"],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="automl autonlp huggingface",
)
