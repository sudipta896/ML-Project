"""
Setup script for the ML project package.
"""

from setuptools import find_packages, setup
from typing import List


def get_requirements(requirements_path: str) -> List[str]:
    '''This function will return the list of requirements'''
    requirements = []
    with open(requirements_path, encoding='utf-8') as file:
        requirements = file.readlines()
        requirements = [req.replace('\n', '') for req in requirements]
        
        if '-e .' in requirements:
            requirements.remove('-e .')
    return requirements


setup(
    name='mlproject',
    version='0.1.0',
    author='sudipta',
    author_email='sudiptamondal86106@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements(requirements_path='requirements.txt')
)



