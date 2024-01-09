from setuptools import setup, find_namespace_packages

setup(name='dynamic_network_architectures',
      packages=find_namespace_packages(include=["dynamic_network_architectures", "dynamic_network_architectures.*"]),
      version='0.2',
      description='none',
      author='Fabian Isensee',
      author_email='f.isensee@dkfz.de',
      license='private',
      install_requires=[
            "torch>=1.6.0a",
            "numpy"
            ],
      zip_safe=False)
