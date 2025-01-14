import json
import os

from setuptools import find_packages, setup


def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(here, rel_path)) as fp:
        return fp.read()


def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith("__version__"):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    raise RuntimeError("Unable to find version string.")


VERSION = get_version("evolve_car/__init__.py")

with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "requirements.txt")) as req_file:
    requirements = req_file.read().splitlines()


CLASSIFIERS = [
    "Development Status :: Alpha",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Operating System :: POSIX :: Linux",
    "Operating System :: Microsoft :: Windows",
    "Topic :: Scientific/Engineering",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Topic :: Software Development",
    "Topic :: Software Development :: Libraries",
    "Topic :: Software Development :: Libraries :: Python Modules",
    "Programming Language :: Python",
    "Programming Language :: Python :: 3 :: Only",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
]

long_description = (
    "EvolveCar: use AI technology to learn how to drive."
)

description = long_description.split(".", maxsplit=1)[0] + "."

setup(
    name="evolve-car",
    version=VERSION,
    description=description,
    long_description=long_description,
    author_email="sygin.li@gmail.com",
    license="MIT License",
    classifiers=CLASSIFIERS,
    url="https://github.com/good-sijin/EvolveCar/",
    download_url="https://github.com/good-sijin/EvolveCar/tags",
    python_requires="==3.8.0",
    install_requires=requirements,
    include_package_data=False,
    data_files=[],
    entry_points={},
    packages=find_packages(exclude=["data"]),
)