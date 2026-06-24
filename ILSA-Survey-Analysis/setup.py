from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="ilsa-survey-analysis",
    version="1.0.0",
    author="ILSA Research Team",
    author_email="",
    description="Structured metadata from ILSA research articles using machine learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/ILSA-Survey-Analysis",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "pyarrow>=14.0.0",
        "pydantic>=2.0.0",
        "python-dotenv>=1.0.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "analysis": [
            "scikit-learn>=1.3.0",
            "matplotlib>=3.7.0",
            "seaborn>=0.12.0",
            "jupyter>=1.0.0",
        ],
        "extraction": [
            "openai>=1.0.0",
            "pymupdf>=1.23.0",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["ilsa_survey_articles/*", "ilsa_survey_articles/json/*.json"],
    },
)