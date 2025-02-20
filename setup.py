from setuptools import find_packages, setup

setup(
    name="unit-tokenizer",
    version="0.1.3",
    author="Shuichiro Shimizu",
    author_email="sshimizu@nlp.ist.i.kyoto-u.ac.jp",
    description="Tokenizers that operate on integer sequences.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/cromz22/unit-tokenizer",
    packages=find_packages(exclude=["tests*", "data*"]),
    include_package_data=True,
    install_requires=[],
    extras_require={
        "dev": [
            "pytest==8.2.1",
            "black==24.4.2",
            "isort==5.13.2",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
)
