from setuptools import find_packages, setup
from typing import List

HYPEN_E_DOT = '-e .'
def get_requirements(file_path: str) -> List[str]:
    """This function returns the list of requirements from the given file."""
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n","") for req in requirements]  # Strip newline and spaces

    # Remove '-e .' if present
    if HYPEN_E_DOT in requirements:
        requirements.remove(HYPEN_E_DOT)

    return requirements

setup(
    name='Krishi-sahayakk',
    version='0.0.1',
    author='Vivek',
    author_email='9999vivekprasad@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirement.txt'),  # Corrected filename
)
