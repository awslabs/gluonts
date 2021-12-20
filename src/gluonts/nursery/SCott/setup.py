from setuptools import setup, find_packages

setup(
    name='pytorchts',
    version='0.2.0',
    description="PyTorch Probabilistic Time Series Modeling framework",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author='Kashif Rasul',
    author_email="kashif.rasul@zalando.de",
    url='https://github.com/zalandoresearch/pytorch-ts',
    license='MIT',

    packages=find_packages(exclude=["tests"]),
    include_package_data=True,
    zip_safe=True,
    python_requires=">=3.6",
    install_requires = [
        'torch>=1.5.0',
        'holidays',
        'numpy',
        'pandas>=1.0,<1.1',
        'scipy',
        'tqdm',
        'pydantic',
        'matplotlib',
        'python-rapidjson',
        'tensorboard',
    ],

    test_suite='tests',
    tests_require = [
        'flake8',
        'pytest'
    ],
)
