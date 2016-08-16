"""
Setup script.
"""

import os
import setuptools


def read(fname):
    """Construct the name and descriptions from README.md."""
    text = open(os.path.join(os.path.dirname(__file__), fname)).read()
    text = text.split('\n\n')
    name = text[0].lstrip('#').strip()
    description = text[1].strip('.')
    long_description = text[2]
    return name, description, long_description


def main():
    """Run the setup."""
    NAME, DESCRIPTION, LONG_DESCRIPTION = read('README.md')
    setuptools.setup(
        name=NAME,
        version='0.0.1',
        author='Matthew W. Hoffman',
        author_email='mwh30@cam.ac.uk',
        url='http://github.com/mwhoffman/' + NAME,
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        license='Simplified BSD',
        packages=setuptools.find_packages(),
        package_data={'': ['*.txt', '*.npz']},
        install_requires=['numpy', 'scipy', 'tabulate', 'ezplot'])


if __name__ == '__main__':
    main()
