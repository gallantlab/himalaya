from setuptools import setup

requirements = [
    "numpy",
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
