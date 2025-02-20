from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="hyperliquid-vault-analyzer",
    version="0.1.0",
    author="StreetJammer",
    description="Advanced ML-powered analyzer for Hyperliquid vaults",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/StreetJammer/hyperliquid-vault-analyzer",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Office/Business :: Financial :: Investment",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.26.4",
        "pandas>=2.2.3",
        "optuna>=4.2.1",
        "scikit-learn",
        "lightgbm>=4.6.0",
        "cvxpy>=1.6.0",
        "hyperliquid-python-sdk>=0.10.0",
        "python-dotenv>=1.0.1",
        "xlsxwriter"
    ],
)
