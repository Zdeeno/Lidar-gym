#!/bin/bash

cd /usr/local
mkdir kitti_dataset
cd kitti_dataset

# download velodyne data
# city
wget http://kitti.is.tue.mpg.de/kitti/raw_data/2011_09_26_drive_0002/2011_09_26_drive_0002_sync.zip
wget http://kitti.is.tue.mpg.de/kitti/raw_data/2011_09_26_drive_0020/2011_09_26_drive_0005_sync.zip
wget http://kitti.is.tue.mpg.de/kitti/raw_data/2011_09_26_drive_0027/2011_09_26_drive_0009_sync.zip
wget http://kitti.is.tue.mpg.de/kitti/raw_data/2011_09_26_drive_0027/2011_09_26_drive_0011_sync.zip
# residential
wget http://kitti.is.tue.mpg.de/kitti/raw_data/2011_09_26_drive_0027/2011_09_26_drive_0019_sync.zip
wget http://kitti.is.tue.mpg.de/kitti/raw_data/2011_09_26_drive_0027/2011_09_26_drive_0020_sync.zip
wget http://kitti.is.tue.mpg.de/kitti/raw_data/2011_09_26_drive_0027/2011_09_26_drive_0022_sync.zip
wget http://kitti.is.tue.mpg.de/kitti/raw_data/2011_09_26_drive_0027/2011_09_26_drive_0039_sync.zip
wget http://kitti.is.tue.mpg.de/kitti/raw_data/2011_09_26_drive_0027/2011_09_26_drive_0061_sync.zip
# road
wget http://kitti.is.tue.mpg.de/kitti/raw_data/2011_09_26_drive_0027/2011_09_26_drive_0027_sync.zip
wget http://kitti.is.tue.mpg.de/kitti/raw_data/2011_09_26_drive_0027/2011_09_26_drive_0015_sync.zip
wget http://kitti.is.tue.mpg.de/kitti/raw_data/2011_09_26_drive_0027/2011_09_26_drive_0070_sync.zip
wget http://kitti.is.tue.mpg.de/kitti/raw_data/2011_09_26_drive_0027/2011_09_26_drive_0016_sync.zip
wget http://kitti.is.tue.mpg.de/kitti/raw_data/2011_09_26_drive_0027/2011_09_26_drive_0047_sync.zip


# download calibration data
wget http://kitti.is.tue.mpg.de/kitti/raw_data/2011_09_26_calib.zip

# extract
for f in *.zip
do
	unzip -o $f
done

# clean up
rm *.zip
