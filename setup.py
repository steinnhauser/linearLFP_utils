import setuptools

setuptools.setup(
    name="masters_utils",
    version="0.0.1",
    author="Steinn Hauser",
    description="Functions for masters thesis.",
    packages=setuptools.find_packages(),
    install_requires=["numpy", "scipy", "argparse"],
    python_requires=">=3.6"
)
