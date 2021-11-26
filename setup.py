from setuptools import setup

setup(name='regexkb',
      version='0.1.0',
      description='Regex Queries over Knowledge Bases',
      url='https://github.com/vaibhavad/KBI-Regex',
      license='MIT',
      packages=['regexkb'],
      install_requires=[
          'numpy>=1.19.0',
          'tqdm>=4.53.0',
          'torch>=1.6.0',
          'pytorch-lightning>=1.1.0',
      ],
      include_package_data=True,
      zip_safe=False)
