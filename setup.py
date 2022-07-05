import setuptools
from pathlib import Path

this_dir = Path(__file__).parent

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="g2p_fa",
    version="1.0.0",
    author="Mohamadhosein Dehghani",
    author_email="demh1377@gmail.com",
    description="A Persian Grapheme to Phoneme model using LSTM implemented in pytorch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/de-mh/g2p_fa",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=["g2p_fa"],
    install_requires=['torch'],
    package_data={'g2p_fa': ['data/*']},
    python_requires=">=3.6",
)