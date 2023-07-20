from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='YourPackageName',
    version='1.0',
    packages=find_packages(),
    install_requires=requirements,
    # 其他元数据，如作者、描述等
)
