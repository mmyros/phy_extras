from setuptools import setup, find_packages

setup(
    name='phy_extras',
    version='0.0.1',
    install_requires=[
        'ssm @ git+ssh://git@github.com/mmyros/ssm.git#egg=ssm',
        'phylib @ git+ssh://git@github.com/cortex-lab/phylib.git#egg=phylib',
        'phy @ git+ssh://git@github.com/cortex-lab/phy.git#egg=phy',
        'cluster_quality @ git+ssh://git@github.com:mmyros/cluster_quality.git#egg=cluster_quality',
    ],
    packages=find_packages(),
    include_package_data=True,
    py_modules=['phy_extras'],
    url='',
    license='MIT',
    author='Maxym Myroshnychenko',
    author_email='mmyros@gmail.com',
    description='extras for spikesorting curation'
)
