from setuptools import setup, find_packages

setup(
    name="marl_optimisation_ctde",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        # List your dependencies here, or rely on requirements.txt
        # e.g. "numpy", "gym", "optuna", etc.
        "tensorflow", "gym", "numpy", "optuna", "pyyaml"
    ],
)
