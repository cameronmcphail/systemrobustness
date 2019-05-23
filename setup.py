"""SystemRobustness calculates and analyses system robustness."""

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="systemrobustness",
    version="0.0.1",
    author="Cameron McPhail",
    author_email="cameron.mcphail@adelaide.edu.au",
    description="Calculations and analyses of system robustness.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/cameronmcphail/systemrobustness",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
