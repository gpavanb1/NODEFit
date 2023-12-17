from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='NODEFit',
      version='0.1',
      description='Fit time-series data with a Neural Differential Equation',
      url='https://github.com/gpavanb1/NODEFit',
      author='gpavanb1',
      author_email='gpavanb@gmail.com',
      license='MIT',
      packages=['nodefit'],
      install_requires=["numpy", "scipy", "torch",
                        "torchdiffeq", "torchsde", "matplotlib"],
      long_description=long_description,
      long_description_content_type="text/markdown",
      classifiers=[
          'Topic :: Scientific/Engineering :: Mathematics',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7',
          'Programming Language :: Python :: 3 :: Only',
      ],
      keywords='python neural-network pytorch numerical-methods neural-ode',
      project_urls={  # Optional
          'Bug Reports': 'https://github.com/gpavanb1/NODEFit/issues',
          'Source': 'https://github.com/gpavanb1/NODEFit/',
      },
      zip_safe=False)
