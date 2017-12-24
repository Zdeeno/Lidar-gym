from setuptools import Extension, find_packages, setup
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))


def file_content(fpath):
    with open(path.join(here, fpath), encoding='utf-8') as f:
        return f.read()


setup(
    name='lidar_gym',
    version='0.0.1',
    description='OpenAI gym training environment for agents controlling solid-state lidars',
    long_description=file_content('README.md'),
    url='https://gitlab.fel.cvut.cz/rozsyzde/lidar-gym',
    author='Zdenek Rozsypalek, CVUT',
    author_email='rozsyzde@fel.cvut.cz',
    license='MIT',
    packages=["tools", "gym_maze.envs"],

    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',

        'Intended Audience :: Information Technology',
        'Intended Audience :: Science/Research',

        'Topic :: Software Development :: Build Tools',
        'Topic :: Multimedia :: Graphics :: 3D Modeling',
        'Topic :: Multimedia :: Graphics :: 3D Rendering',

        'License :: OSI Approved :: MIT License',

        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
    keywords='gym lidar environment openai',
    install_requires=['numpy', 'pykitty', 'voxel_map', 'gym'],
)
