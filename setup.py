from setuptools import setup, find_packages

python_requires = ">=3.8"
required_python_version = (3, 8)

with open("README.md") as f:
    LONG_DESCRIPTION = f.read()

metadata = dict(
    name="mcba",
    maintainer="Maicon Dallagnol",
    maintainer_email="maicon.dallagnol@unesp.br",
    description="A Python package for Keel datasets processed for Machine Learning experiments.",
    license="BSD",
    url="https://github.com/maicondallg/KeelDS",
    download_url="https://github.com/maicondallg/KeelDS",
    project_urls={
        "Bug Tracker": "https://github.com/maicondallg/KeelDS/issues",
        "Source Code": "https://github.com/maicondallg/KeelDS",
    },
    package_data={"keel_ds": "data/*"},
    include_package_data=True,
    version="0.1.0",
    long_description=LONG_DESCRIPTION,
    python_requires=">=3.8",
    install_requires=["numpy"],
    zip_safe=False,
    packages=find_packages(),
)

setup(**metadata)
