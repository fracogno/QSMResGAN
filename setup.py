from setuptools import setup, find_packages

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='QSMResGAN',
    version='2.0.0',
    description='Implementation with TF 2.0',
    long_description=readme,
    author='Francesco Cognolato',
    author_email='francesco.cognolato@gmail.com',
    url='https://github.com/fracogno/QSMResGAN/',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)
