from setuptools import setup

setup(
    # Needed to silence warnings (and to be a worthwhile package)
    name='wings',
    url='https://github.com/Rubyr0id/wings',
    author='Anton Dubrovitskiy',
    author_email='adubrovitskiy@gmail.com',
    # Needed to actually package something
    packages=['wings'],
    # Needed for dependencies
    install_requires=['numpy','pandas',''],
    # *strongly* suggested for sharing
    version='1.1',
    # The license can be anything you like
    license='MIT',
    description='WOE transformation',
    # We will also need a readme eventually (there will be a warning)
    long_description = open('README.txt').read(),
)