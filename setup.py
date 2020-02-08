requirements = []
with open('requirements.txt', 'r') as fh:
    for line in fh:
        requirements.append(line.strip())

from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='Skperopt',
    version='0.0.73',
    packages=["skperopt"],
    url='https://github.com/lewis-morris/Skperopt',
    license='MIT',
    author='Lewis',
    description='A hyperopt wrapper - simplifying hyperparameter tuning with Scikit-learn style estimators.',
    author_email='lewis.morris@gmail.com',
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=requirements,
    download_url = 'https://github.com/lewis-morris/Skperopt/archive/0.0.73.tar.gz',
    keywords=['hyperopt-wrapper', 'hyperparameter'],
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Intended Audience :: Information Technology',
        'Intended Audience :: System Administrators',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Natural Language :: English',
    ],
)
