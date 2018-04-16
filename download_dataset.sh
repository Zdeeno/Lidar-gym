#!/bin/bash

cd ~
mkdir kitti_dataset
cd kitti_dataset

# download velodyne data
# city
wget http://kitti.is.tue.mpg.de/kitti/raw_data/2011_09_26_drive_0001/2011_09_26_drive_0001_sync.zip
wget http://kitti.is.tue.mpg.de/kitti/raw_data/2011_09_26_drive_0002/2011_09_26_drive_0002_sync.zip
wget http://kitti.is.tue.mpg.de/kitti/raw_data/2011_09_26_drive_0005/2011_09_26_drive_0005_sync.zip
wget http://kitti.is.tue.mpg.de/kitti/raw_data/2011_09_26_drive_0009/2011_09_26_drive_0009_sync.zip # VALIDATION
wget http://kitti.is.tue.mpg.de/kitti/raw_data/2011_09_26_drive_0011/2011_09_26_drive_0011_sync.zip
wget http://kitti.is.tue.mpg.de/kitti/raw_data/2011_09_26_drive_0013/2011_09_26_drive_0013_sync.zip
wget http://kitti.is.tue.mpg.de/kitti/raw_data/2011_09_26_drive_0014/2011_09_26_drive_0014_sync.zip
wget http://kitti.is.tue.mpg.de/kitti/raw_data/2011_09_26_drive_0017/2011_09_26_drive_0017_sync.zip
wget http://kitti.is.tue.mpg.de/kitti/raw_data/2011_09_26_drive_0018/2011_09_26_drive_0018_sync.zip
wget http://kitti.is.tue.mpg.de/kitti/raw_data/2011_09_26_drive_0048/2011_09_26_drive_0048_sync.zip
wget http://kitti.is.tue.mpg.de/kitti/raw_data/2011_09_26_drive_0051/2011_09_26_drive_0051_sync.zip
wget http://kitti.is.tue.mpg.de/kitti/raw_data/2011_09_26_drive_0056/2011_09_26_drive_0056_sync.zip
wget http://kitti.is.tue.mpg.de/kitti/raw_data/2011_09_26_drive_0057/2011_09_26_drive_0057_sync.zip
wget http://kitti.is.tue.mpg.de/kitti/raw_data/2011_09_26_drive_0059/2011_09_26_drive_0059_sync.zip
wget http://kitti.is.tue.mpg.de/kitti/raw_data/2011_09_26_drive_0060/2011_09_26_drive_0060_sync.zip
wget http://kitti.is.tue.mpg.de/kitti/raw_data/2011_09_26_drive_0084/2011_09_26_drive_0084_sync.zip
wget http://kitti.is.tue.mpg.de/kitti/raw_data/2011_09_26_drive_0091/2011_09_26_drive_0091_sync.zip
wget http://kitti.is.tue.mpg.de/kitti/raw_data/2011_09_26_drive_0093/2011_09_26_drive_0093_sync.zip
wget http://kitti.is.tue.mpg.de/kitti/raw_data/2011_09_26_drive_0095/2011_09_26_drive_0095_sync.zip
wget http://kitti.is.tue.mpg.de/kitti/raw_data/2011_09_26_drive_0096/2011_09_26_drive_0096_sync.zip
wget http://kitti.is.tue.mpg.de/kitti/raw_data/2011_09_26_drive_0104/2011_09_26_drive_0104_sync.zip


# --- using only city dataset
# residential
# wget http://kitti.is.tue.mpg.de/kitti/raw_data/2011_09_26_drive_0019/2011_09_26_drive_0019_sync.zip
# wget http://kitti.is.tue.mpg.de/kitti/raw_data/2011_09_26_drive_0020/2011_09_26_drive_0020_sync.zip
# wget http://kitti.is.tue.mpg.de/kitti/raw_data/2011_09_26_drive_0022/2011_09_26_drive_0022_sync.zip
# wget http://kitti.is.tue.mpg.de/kitti/raw_data/2011_09_26_drive_0039/2011_09_26_drive_0039_sync.zip
# wget http://kitti.is.tue.mpg.de/kitti/raw_data/2011_09_26_drive_0061/2011_09_26_drive_0061_sync.zip
# road
# wget http://kitti.is.tue.mpg.de/kitti/raw_data/2011_09_26_drive_0027/2011_09_26_drive_0027_sync.zip
# wget http://kitti.is.tue.mpg.de/kitti/raw_data/2011_09_26_drive_0015/2011_09_26_drive_0015_sync.zip
# wget http://kitti.is.tue.mpg.de/kitti/raw_data/2011_09_26_drive_0070/2011_09_26_drive_0070_sync.zip
# wget http://kitti.is.tue.mpg.de/kitti/raw_data/2011_09_26_drive_0029/2011_09_26_drive_0029_sync.zip
# wget http://kitti.is.tue.mpg.de/kitti/raw_data/2011_09_26_drive_0032/2011_09_26_drive_0032_sync.zip


# download calibration data
wget http://kitti.is.tue.mpg.de/kitti/raw_data/2011_09_26_calib.zip

# extract
for f in *.zip
do
	unzip -o $f
	rm $f                                       # remove zip files
	find . -name "*.png" -type f -delete        # remove pngs (a lot of bytes on HDD)
done

# repair drive 0009 with corrupted dataset
cd 2011_09_26/2011_09_26_drive_0009_sync/oxts/data
rm 0000000177.txt
rm 0000000178.txt
rm 0000000179.txt
rm 0000000180.txt
cd ../../../..

echo 'Now you can serialize your dataset using python -m lidar_gym.tools.map_parser -s'
