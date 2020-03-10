from setuptools import setup

requirements = [
    "numpy",
    "scikit-learn",
    # "scipy", # optional
    # "cupy",  # optional
    # "torch",  # optional
    # "matplotlib",  # for visualization only
    # "pytest",  # for testing only
]

if __name__ == "__main__":
    setup(
        name='himalaya',
        version='0.1',
        packages=[
            'himalaya',
        ],
        install_requires=requirements,
    )
