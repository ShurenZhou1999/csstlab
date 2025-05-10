from setuptools import setup, find_packages

setup(
    name="csstlab",
    version="1.0",
    packages=find_packages(), 
    package_data={
        "my_package": ["data/*.npy"], 
    },
    install_requires=[ 
        "numpy",
        "scipy",
    ],
    description="CSST Hybrid Lagrangian Bias Expansion Emulator", 
    long_description=open("README.md").read(),
    url="https://github.com/ShurenZhou1999/csstlab",
    author="Shuren Zhou", 
    author_email="zhoushuren@sjtu.edu.cn",  
    license="MIT", 
)