requirements = []
with open('requirements.txt', 'r') as fh:
    for line in fh:
        requirements.append(line.strip())

from setuptools import setup

setup(
    name='Skperopt',
    version='0.0.5',
    packages=["skperopt"],
    url='https://github.com/lewis-morris/Skperopt',
    license='MIT',
    author='Lewis',
    author_email='lewis.morris@gmail.com',
    description='Hyperopt Wrapper',
    install_requires=requirements,
    download_url = 'https://github.com/lewis-morris/Skperopt/archive/0.0.5.tar.gz',
    keywords=['hyperopt-wrapper', 'hyperparameter'],
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Intended Audience :: Information Technology',
        'Intended Audience :: System Administrators',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Natural Language :: English',
    ],
)
