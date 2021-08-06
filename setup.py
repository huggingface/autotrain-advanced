# Lint as: python3
""" HuggingFace/AutoNLP
"""
import os

from setuptools import find_packages, setup


DOCLINES = __doc__.split("\n")

this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, "README.md"), encoding="utf-8") as f:
    LONG_DESCRIPTION = f.read()

INSTALL_REQUIRES = [
    "loguru==0.5.3",
    "requests==2.25.1",
    "tqdm==4.49",
    "prettytable==2.0.0",
    "huggingface_hub==0.0.12",
    "datasets==1.11.0",
]

QUALITY_REQUIRE = [
    "black",
    "isort",
    "flake8==3.7.9",
]

TESTS_REQUIRE = ["pytest"]


EXTRAS_REQUIRE = {
    "dev": INSTALL_REQUIRES + QUALITY_REQUIRE + TESTS_REQUIRE,
    "quality": INSTALL_REQUIRES + QUALITY_REQUIRE,
    "docs": INSTALL_REQUIRES
    + [
        "recommonmark",
        "sphinx==3.1.2",
        "sphinx-markdown-tables",
        "sphinx-rtd-theme==0.4.3",
        "sphinx-copybutton",
    ],
}

setup(
    name="autonlp",
    description=DOCLINES[0],
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    author="HuggingFace Inc.",
    author_email="abhishek@huggingface.co",
    url="https://github.com/huggingface/autonlp",
    download_url="https://github.com/huggingface/autonlp/tags",
    license="Apache 2.0",
    package_dir={"": "src"},
    packages=find_packages("src"),
    scripts=["scripts/autonlp"],
    extras_require=EXTRAS_REQUIRE,
    install_requires=INSTALL_REQUIRES,
    entry_points={"console_scripts": ["autonlp=autonlp.cli.autonlp:main"]},
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
