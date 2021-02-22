# Lint as: python3
""" HuggingFace/AutoNLP
"""

from setuptools import find_packages, setup


DOCLINES = __doc__.split("\n")

INSTALL_REQUIRES = ["loguru==0.5.3", "requests==2.25.1", "tqdm==4.56.0", "prettytable==2.0.0"]

QUALITY_REQUIRE = [
    "black",
    "isort",
    "flake8==3.7.9",
]


EXTRAS_REQUIRE = {
    "dev": INSTALL_REQUIRES + QUALITY_REQUIRE,
    "quality": INSTALL_REQUIRES + QUALITY_REQUIRE,
    "docs": [
        "recommonmark",
        "sphinx==3.1.2",
        "sphinx-markdown-tables",
        "sphinx-rtd-theme==0.4.3",
        "sphinx-copybutton",
    ],
}

setup(
    name="autonlp",
    version="0.0.3",
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
