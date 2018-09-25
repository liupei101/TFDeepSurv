from setuptools import setup, find_packages

with open('README.txt') as file:
    long_description = file.read()

setup(name='TFDeepSurv',
    version='0.0.3',
    description='Deep Cox Proportional Hazards Network implemented by tensorflow framework and survival analysis.',
    keywords = "survival analysis deep learning cox regression tensorflow",
    url='https://github.com/liupei101/TFDeepSurv',
    author='Pei Liu',
    author_email='18200144374@163.com',
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
        'tensorflow>=1.4.0',
        'scikit-learn>=0.19.0',
        'lifelines>=0.9.2',
        'hyperopt==0.1',
        'supersmoother>=0.4',
    ],
)