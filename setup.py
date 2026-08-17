from setuptools import Extension, find_packages, setup

ext_modules = [
    Extension(
        name="cminbpe._minbpe",  # put the binary in the cminbpe package
        sources=["cminbpe/_minbpe.c"],
        include_dirs=["cminbpe"],
        extra_compile_args=["-O3"],
    )
]

setup(
    name="cminbpe",
    version="0.1.0",
    description="A minimal BPE tokenizer implementation with C backend for fast training and tokenization.",
    author="abiget",
    author_email="getachewante@gmail.com",
    packages=find_packages(include=["cminbpe"]),
    ext_modules=ext_modules,
)
