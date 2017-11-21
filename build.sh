#!/bin/bash

rm -r build
mkdir build
cd build
cmake ..
make
echo import voxel_map > __init__.py
cd ..
