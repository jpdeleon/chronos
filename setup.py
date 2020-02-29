"""
This installation requires git which pulls large files first before pip installation
https://stackoverflow.com/a/58932741/1910174
"""
from setuptools import setup, find_packages

import os
import subprocess
import sys

# try:
#     import git
# except ModuleNotFoundError:
#     subprocess.call([sys.executable, "-m", "pip", "install", "gitpython"])
#     import git


def install_requires():
    reqdir = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(reqdir, "requirements.txt"), encoding="utf-8") as f:
        all_packages = f.readlines()
        packages = [
            package for package in all_packages if "git+ssh" not in package
        ]
        manual_pip_packages = [
            package for package in all_packages if "git+ssh" in package
        ]
        for package in manual_pip_packages:
            subprocess.call([sys.executable, "-m", "pip", "install", package])
    return packages


# def pull_first():
#     """This script is in a git directory that can be pulled."""
#     cwd = os.getcwd()
#     gitdir = os.path.dirname(os.path.realpath(__file__))
#     os.chdir(gitdir)
#     g = git.cmd.Git(gitdir)
#     try:
#         g.execute(["git", "lfs", "pull"])
#     except git.exc.GitCommandError:
#         raise RuntimeError("Make sure git-lfs is installed!")
#     os.chdir(cwd)
#
#
# pull_first()

setup(
    name="chronos",
    version="0.0.1",
    description="exploring young star catalogs",
    url="http://github.com/jpdeleon/chronos",
    author="Jerome de Leon",
    author_email="jpdeleon.bsap@gmail.com",
    license="MIT",
    packages=["chronos"],  # or find_packages(),
    include_package_data=True,
    # data_files=['data'],
    # package_data={"chronos": "data"},
    scripts=[
        "scripts/make_tql",
        "scripts/make_tql_per_cluster",
        "scripts/query_cluster_members_gaia_params",
        "scripts/find_cluster_near_target",
        "scripts/make_cdips_ql",
    ],
    zip_safe=False,
    install_requires=install_requires(),
)

print("Finally, git lfs pull")
