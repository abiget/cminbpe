from setuptools import Extension, find_packages, setup

ext_modules = [
    Extension(
        name="cminbpe._minbpe",  # put the binary in the cminbpe package
        sources=["cminbpe/_minbpe.c"],
    )
]

setup(
    name="cminbpe",
    version="0.1.0",
    description="C implementation of BPE tokenizer",
    author="abiget",
    author_email="getachewante@gmail.com",
    packages=find_packages(include=["cminbpe"]),
    ext_modules=ext_modules,
    cflags=["-O3", "-march=native"],
)
