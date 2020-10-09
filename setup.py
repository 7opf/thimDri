from setuptools import setup, find_packages

with open("README.md", "r") as readme_file:
    readme = readme_file.read()

requirements = ["numpy>=1.19", "pandas>=1.1"]

setup(
    name="thim-dri",
    version="0.1.0",
    author="Eyal Soreq",
    author_email="eyalsoreq@gmail.com",
    description="A package to process thim iot datasets",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/esoreq/thim-dri.git",
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3.8.5",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
    ],
)
