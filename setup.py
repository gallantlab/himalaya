import re
from setuptools import find_packages, setup

# get version from himalaya/__init__.py
with open('himalaya/__init__.py') as f:
    infos = f.readlines()
for line in infos:
    if "__version__" in line:
        match = re.search(r"__version__ = '([^']*)'", line)
        __version__ = match.groups()[0]

requirements = [
    "numpy",
    "scikit-learn",
    # "cupy",  # optional
    # "torch",  # optional
    # "matplotlib",  # for visualization only
    # "pytest",  # for testing only
]

if __name__ == "__main__":
    setup(
        name='himalaya',
        maintainer="Tom Dupre la Tour",
        maintainer_email="tom.dupre-la-tour@m4x.org",
        description="Multiple-target machine learning",
        license='BSD (3-clause)',
        version=__version__,
        packages=find_packages(),
        url="https://github.com/tomdlt/himalaya",
        install_requires=requirements,
    )
