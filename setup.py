import re
from pathlib import Path
from setuptools import find_packages, setup

# get version from himalaya/__init__.py
__version__ = 0.0
with open('himalaya/__init__.py') as f:
    infos = f.readlines()
for line in infos:
    if "__version__" in line:
        match = re.search(r"__version__ = '([^']*)'", line)
        __version__ = match.groups()[0]

# read the contents of the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.rst").read_text()

requirements = [
    "numpy",
    "scikit-learn",
    # "cupy",  # optional backend
    # "torch",  # optional backend
    # "matplotlib",  # for visualization only
    # "pytest",  # for testing only
]

extras_require = {
    "all_backends": ["cupy", "torch"],
    "viz": ["matplotlib"],
    "test": ["pytest", "cupy", "torch"],
}

extras_require["doc"] = sum(list(extras_require.values()), [])

if __name__ == "__main__":
    setup(
        name='himalaya',
        maintainer="Tom Dupre la Tour",
        maintainer_email="tomdlt@berkeley.edu",
        description="Multiple-target machine learning",
        license='BSD (3-clause)',
        version=__version__,
        packages=find_packages(),
        url="https://github.com/gallantlab/himalaya",
        install_requires=requirements,
        extras_require=extras_require,
        long_description=long_description,
        long_description_content_type='text/x-rst',
    )
