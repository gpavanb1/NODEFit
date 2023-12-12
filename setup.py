from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='FContin',
      version='0.1',
      description='Bayesian (Physics-Informed) Neural Differential Equations in Python',
      url='https://github.com/gpavanb1/BPNets',
      author='gpavanb1',
      author_email='gpavanb@gmail.com',
      license='MIT',
      packages=['bpnets'],
      install_requires=["numpy",
                        ],
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
      keywords='python bayesian numerical ML',
      project_urls={  # Optional
          'Bug Reports': 'https://github.com/gpavanb1/BPNets/issues',
          'Source': 'https://github.com/gpavanb1/BPNets/',
      },
      zip_safe=False)
