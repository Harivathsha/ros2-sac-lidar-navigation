from glob import glob
from setuptools import find_packages, setup

setup(
    name="nav_learning", version="1.0.0", packages=find_packages(exclude=["test"]),
    data_files=[("share/ament_index/resource_index/packages", ["resource/nav_learning"]),
                ("share/nav_learning", ["package.xml"]),
                ("share/nav_learning/config", glob("config/*.yaml"))],
    install_requires=["setuptools"], zip_safe=True,
    maintainer="harivathsha", maintainer_email="harivathsharamesh@gmail.com",
    description="SAC steering navigation for the original Jazzy square arena",
    license="Apache-2.0",
    entry_points={"console_scripts": [
        "train_sac = nav_learning.train_sac:main",
        "evaluate_sac = nav_learning.evaluate_sac:main",
        "check_sac = nav_learning.check_sac:main",
        "plot_sac_results = nav_learning.plot_sac_results:main"]},
)
