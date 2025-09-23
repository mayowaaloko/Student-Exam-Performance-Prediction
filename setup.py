from setuptools import find_packages, setup

hyphen_e_dot = "-e ."


def get_requirements(file_path: str) -> list[str]:
    """
    this function will return the list of requirements
    """
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n", "") for req in requirements]

        if hyphen_e_dot in requirements:
            requirements.remove(hyphen_e_dot)
    return requirements


setup(
    name="mlproject",
    packages=find_packages(),
    version="0.0.1",
    author="mayowa",
    author_email="mayowaaloko@gmail.com",
    install_requires=get_requirements("requirements.txt"),
)
