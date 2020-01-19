from setuptools import setup

setup(
    name='Skperopt',
    version='0.01',
    packages=['skperopt'],
    url='https://github.com/lewis-morris/Skperopt',
    license='MIT',
    author='Lewis',
    author_email='lewis.morris@gmail.com',
    description='Hyperopt Wrapper',
    instal_requires=['numpy', 'pandas', 'sklearn', 'hyperopt>=0.2.3'],
    download_url = 'https://github.com/lewis-morris/Skperopt/archive/0.02.tar.gz'
)
