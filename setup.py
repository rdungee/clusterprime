from setuptools import setup

setup(name='clusterprime',
      version='1.0.0',
      description='turning MEGAPRIME data into Light Curves',
      url='https://github.com/rdungee/clusterprime',
      author='Ryan Dungee',
      author_email='rdungee@hawaii.edu',
      license='MIT',
      packages=['clusterprime'],
      zip_safe=False,
      install_requires=['numpy >= 1.19.5',
                        'astropy >= 4.1',
                        'photutils >= 1.0.2',
                        'scipy >= 1.5.4'])
