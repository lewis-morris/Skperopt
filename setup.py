from setuptools import setup

setup(
    name='Skperopt',
    version='0.02',
    packages=setup.find_packages(),
    url='https://github.com/lewis-morris/Skperopt',
    license='MIT',
    author='Lewis',
    author_email='lewis.morris@gmail.com',
    description='Hyperopt Wrapper',
    instal_requires=['numpy', 'pandas', 'sklearn', 'hyperopt>=0.2.3'],
    download_url = 'https://github.com/lewis-morris/Skperopt/archive/0.0.1.tar.gz',
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
