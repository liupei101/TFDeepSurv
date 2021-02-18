#coding=utf-8
from setuptools import setup, find_packages

pkg_name = "tfdeepsurv"


exec(compile(open(pkg_name+"/version.py").read(), pkg_name+"/version.py", "exec"))


with open("README.md") as f:
    long_description = f.read()

setup(name=pkg_name,
    version=__version__,
    description='Deep cox proportional hazards model implemented by tensorflow framework and survival analysis.',
    keywords = "survival analysis, deep learning, cox regression, tensorflow",
    url='https://github.com/liupei101/TFDeepSurv',
    author='Pei Liu',
    author_email='yuukilp@163.com',
    license='MIT',
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "License :: MIT License",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering",
    ],
    packages = find_packages(),
    python_requires=">=3.5",
    install_requires=[
        'tensorflow>=1.14.0, <2.0.0',
        'pandas>=0.24.2',
        'numpy>=1.14.5',
        'matplotlib>=3.0.3',
        'lifelines>=0.14.6',
        'scikit-learn'
    ],
    include_package_data=True,
)