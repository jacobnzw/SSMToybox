from setuptools import setup, find_packages


setup(
    name='ssmtoybox',
    version='0.1.1a0',
    url='https://github.com/jacobnzw/SSMToybox/tree/v0.1.1-alpha',
    license='MIT',
    author='Jakub Prüher',
    author_email='jacobnzw@gmail.com',
    description='Local filters based on Bayesian quadrature',
    long_description=open('README.md').read(),
    packages=find_packages(),
    zip_safe=False,
    setup_requires=['nose>=1.0'],
    test_suite='nose.collector'
)