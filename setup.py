import setuptools

with open('README.md') as file:
    ld = file.read()

setuptools.setup(
    name='ssmtoybox',
    version='0.1.1a0',
    url='https://github.com/jacobnzw/SSMToybox/tree/v0.1.1-alpha',
    license='MIT',
    author='Jakub Prüher',
    author_email='jacobnzw@gmail.com',
    description='Local filters based on Bayesian quadrature',
    long_description=ld,
    long_description_content_type='text/markdown',
    packages=setuptools.find_packages(),
    zip_safe=False,
    setup_requires=['nose>=1.0'],
    test_suite='nose.collector',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Development Status :: 3 - Alpha'
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
    ]
)