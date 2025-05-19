from setuptools import setup, find_packages

setup(
    name="foundation_agent",
    version="0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[],
    author="Tran Van Tuan",
    description="Method and device for foundation agents",
)