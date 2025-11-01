from setuptools import setup, find_packages

def read_requirements(filename):
    with open(filename) as f:
        requirement_l = [req.replace("\n", "") for req in f.readlines() if "\n" in req]
        if "-e ." in requirement_l:
            requirement_l.remove("-e .")
    return requirement_l

setup(
    name="project01",
    version="0.1",
    packages=find_packages(),
    install_requires=read_requirements('requirements.txt'),
    author="Your Name",
    author_email="your.email@example.com",
    description="A short description of your project",
    
    url="https://github.com/halfscientific/project01",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    
)