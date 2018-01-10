#!/bin/bash

DIR=$(dirname "$0")
cd $DIR
cd lidar_gym
mkdir dataset
cd dataset

# download velodyne data
wget http://kitti.is.tue.mpg.de/kitti/raw_data/2011_09_26_drive_0002/2011_09_26_drive_0002_sync.zip
wget http://kitti.is.tue.mpg.de/kitti/raw_data/2011_09_26_drive_0020/2011_09_26_drive_0020_sync.zip
wget http://kitti.is.tue.mpg.de/kitti/raw_data/2011_09_26_drive_0027/2011_09_26_drive_0027_sync.zip

# download calibration data
wget http://kitti.is.tue.mpg.de/kitti/raw_data/2011_09_26_calib.zip

# extract
for f in *.zip
do
	unzip -o $f
done

# clean up
rm *.zip
