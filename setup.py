from setuptools import setup, find_packages

with open('README.txt') as file:
    long_description = file.read()

setup(name='TFDeepSurv',
    version='2.0.0',
    description='Deep Cox Proportional Hazards Network implemented by tensorflow framework and survival analysis.',
    keywords = "survival analysis deep learning cox regression tensorflow",
    url='https://github.com/liupei101/TFDeepSurv',
    author='Pei Liu',
    author_email='yuukilp@163.com',
    license='MIT',
    long_description = long_description,
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python 3",
        "Topic :: Scientific/Engineering",
    ],
    packages = find_packages(),
    install_requires=[
        'tensorflow>=1.10.0',
        'pandas>=0.24.2',
        'numpy>=1.14.5',
        'matplotlib>=3.0.3',
        'lifelines>=0.14.6'
    ],
)