from setuptools import setup, find_packages

setup(
    name='nashypy',
    version='0.0.1',
    packages=find_packages(),
    package_data={'nashypy': ['games/*.txt']},
    install_requires=["plotly", "numpy"],
    test_suite='tests',
    author='Son Pham',
    author_email='snpham02@gmail.com',
    description='Game Theory Library',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/snpham/nashypy',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
