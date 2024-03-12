from setuptools import find_packages,setup
from typing import List

editable_package = '-e .'

def get_requirements(file_path:str)->List[str]:
    '''
    Function to get and return the list of requirements.
    '''
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("\n","") for req in requirements]

        if editable_package in requirements:
            requirements.remove(editable_package)
  
    return requirements



setup(
name='mlproject',
version='0.0.1',
author='Affaan',
author_email='affaan19@gmail.com',
packages=find_packages(),
install_requires=get_requirements('requirements.txt'),
)