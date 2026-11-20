import setuptools


setuptools.setup(
    name="coma",
    version="0.1.0",
    description=(
        "Concept-Enhanced Multi-Scale Cross-Modal Alignment "
        "for Medical Visual Representation Learning (CoMA)"
    ),
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Xiangmin Kong",
    author_email="B202274020@emails.bjut.edu.cn",
    url="https://github.com/RyaoXD/CoMA",
    packages=setuptools.find_packages(),
    python_requires=">=3.8",
    install_requires=[],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
)
