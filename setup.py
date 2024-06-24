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
name='LT-seq2seq',
version='0.1.0',
author='Affaan',
author_email='affaan2k2@gmail.com',
packages=find_packages(),
install_requires=get_requirements('requirements.txt'),
)