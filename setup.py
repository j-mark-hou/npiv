from setuptools import setup, find_packages

setup(
    name='npiv',
    version='0.1',
    maintainer='J. Mark Hou',
    maintainer_email='jmarkhou@jmarkhou.com',
    packages=find_packages(),
    install_requires=[
    'pandas',
    'scipy',
    'statsmodels',
    'scikit-learn',
    'pytest',
    'lightgbm',
    'seaborn'
    ]
    )