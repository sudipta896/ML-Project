"""
Setup script for the ML project package.
"""

from setuptools import find_packages, setup
from typing import List


def get_requirements(requirements_path: str) -> List[str]:
    """Return cleaned list of requirements"""
    requirements = []
    with open(requirements_path, encoding='utf-8') as file:
        requirements = file.readlines()

        # ✅ Clean lines properly (remove spaces, newline)
        requirements = [req.strip() for req in requirements]

        # ✅ Remove empty lines and editable install
        requirements = [
            req for req in requirements 
            if req and req != "-e ."
        ]

    return requirements


setup(
    name="mlproject",
    version="0.1.0",
    author="sudipta",
    author_email="sudiptamondal86106@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt"),
)