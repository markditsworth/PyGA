import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pyga",
    version="0.0.1",
    author="Mark Ditsworth",
    author_email="markditsworth@protonmail.com",
    description="A package for genetic algorithm optimization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/markditsworth/PyGA",
    packages=setuptools.find_packages(),
    classifiers=(
        "Programming Language :: Python :: 2",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
)
