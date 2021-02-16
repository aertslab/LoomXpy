from setuptools import setup, find_packages

def read_requirements(fname):
    with open(fname, 'r', encoding='utf-8') as file:
        return [line.rstrip() for line in file]


setup(
    name='loomxpy',
    use_scm_version=True,
    setup_requires=['setuptools_scm'],
    packages=find_packages(),
    description='Python package (compatible with SCope) to create .loom files and extend them with other data e.g.: SCENIC regulons',
    long_description=open('README.rst').read(),
    url='https://github.com/aertslab/LoomXpy',
    version='0.1.0',
    license='GPL-3.0+',
    author='Maxime De Waegeneer, Kristofer Davie',
    install_requires=read_requirements('requirements.txt'),
    platforms=['any'],
    keywords=['SCope', 'loom', 'single-cell'],
)

