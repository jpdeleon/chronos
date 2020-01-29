from setuptools import setup, find_packages

setup(
    name="chronos",
    version="0.0.1",
    description="exploring young star catalogs",
    url="http://github.com/jpdeleon/chronos",
    author="Jerome de Leon",
    author_email="jpdeleon.bsap@gmail.com",
    license="MIT",
    packages=["chronos"],  # or find_packages(),
    # package_data={"chronos": "data"},
    include_package_data=True,
    scripts=[
        "scripts/query_cluster_members_gaia_params",
        "scripts/find_cluster_near_target",
        "scripts/make_cdips_ql",
    ],
    zip_safe=False,
    install_requires=[
        "astroquery",
        "lightkurve",
        "astropy",
        "pandas",
        "tqdm",
        "transitleastsquares",
        "wotan",
        "deepdish",
    ],
)
