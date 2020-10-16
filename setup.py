
from setuptools import setup, find_packages

with open("README.md") as f:
    readme = f.read()

with open("LICENSE") as f:
    license = f.read()

setup(
    name="npsv2",
    version="0.1.0",
    description="Non-parametric genotyper for structural variants",
    long_description=readme,
    author="Michael Linderman",
    author_email="mlinderman@middlebury.edu",
    license=license,
    url="https://github.com/mlinderm/npsv2",
    scripts=[],
    entry_points="""
        [console_scripts]
        npsv2=npsv2.main:main
    """,
    packages=find_packages("src"),
    package_dir={"": "src"},
    zip_safe=False,
    test_suite="tests",
    include_package_data=True,
    data_files=[
        (
            "etc",
            [
                "etc/human_g1k_v37.genome",
                "etc/human_g1k_v37.gaps.bed.gz",
                "etc/human_g1k_v37.gaps.bed.gz.tbi",
            ],
        )
    ],
    classifiers=[
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.6",
    ],
)
