"""
This installation requires git which pulls large files first before pip installation
https://stackoverflow.com/a/58932741/1910174
"""
from setuptools import setup, find_packages

setup(
    name="chronos",
    version="0.1.0",
    description="discovery and characterization of young stars",
    long_description=open("README.md").read(),
    url="http://github.com/jpdeleon/chronos",
    author="Jerome de Leon",
    author_email="jpdeleon.bsap@gmail.com",
    license="MIT",
    packages=find_packages(),
    # scripts=[
    #     "scripts/check_target_in_cluster",
    #     "scripts/make_tql_per_cluster",
    #     "scripts/query_cluster_members_gaia_params",
    #     "scripts/find_cluster_near_target",
    #     "scripts/make_cdips_ql",
    # ],
    install_requires=[
        "tqdm",
        "astroquery==0.4",
        "lightkurve==1.9.0",
        "astropy==4.0",
        "pandas==1.0.1",
        "astroplan==0.6",
        "transitleastsquares",
        "wotan==1.7",
        "scikit-image==0.16.2",  # just for measure.find_contours
        "deepdish==0.3.6",
    ],
    extras_requires={
        "contaminante": [
            "git+https://github.com/christinahedges/contaminante#egg=contaminante"
        ],
        "popstar": [
            "git+https://github.com/astropy/PyPopStar.git#egg=PyPopStar"
        ],
        "isochrones": [
            "git+https://github.com/timothydmorton/isochrones.git#egg=isochrones"
        ],
        "stardate": [
            "git+https://github.com/RuthAngus/stardate.git#egg=stardate"
        ],
        "dustmaps": [
            "git+https://github.com/gregreen/dustmaps.git#egg=dustmaps"
        ],
        "corner": ["git+https://github.com/dfm/corner.py.git#egg=corner"],
        "maelstrom": [
            "git+https://github.com/danhey/maelstrom.git#egg=maelstrom"
        ],
        "fleck": ["git+https://github.com/bmorris3/fleck.git#egg=fleck"],
    },
)
