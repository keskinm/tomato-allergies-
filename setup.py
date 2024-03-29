from setuptools import setup, find_packages

setup(
    name='fashion-mnist',
    packages=find_packages(),
    zip_safe=True,
    install_requires=[
        'cycler==0.10.0',
        'decorator==4.4.1',
        'imageio==2.6.1',
        'imgaug==0.3.0',
        'kiwisolver==1.1.0',
        'matplotlib==3.1.1',
        'networkx==2.4',
        'numpy==1.17.4',
        'opencv-contrib-python==4.1.1.26',
        'Pillow==6.2.1',
        'pyparsing==2.4.5',
        'python-dateutil==2.8.1',
        'PyWavelets==1.1.1',
        'scikit-image==0.16.2',
        'scipy==1.3.2',
        'six==1.13.0',
    ],
)

