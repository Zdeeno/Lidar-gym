# Lidar gym
OpenAI gym training environment for agents controlling solid-state lidars based on KITTI dataset.
## Dependencies
Environment is written in python 3 using following libraries:
### Required:

##### 1. Numpy
https://github.com/numpy/numpy
##### 2. OpenAI gym
https://github.com/openai/gym
##### 3. Pykitti
https://github.com/utiasSTARS/pykitti
##### 4. Voxel map
https://bitbucket.org/tpetricek/voxel_map

### Optional:
##### 1. Mayavi with VTK
https://github.com/enthought/mayavi (used for visualisation)

##### 2. Tensorflow
https://github.com/tensorflow/tensorflow (used for agents)

## Installation 
Package can be easily installed in terminal by command:
`python setup.py install`<br />
Before running your code, you must download the Kitti dataset.
There is script [download_dataset.sh](download_dataset.sh) which will download the dataset. 
Do **not** change the destination folder of the dataset. Lidar-gym is looking for the dataset in home directory.
There is also possibility to serialize the dataset using **Pickle**. If you want to check that the dataset was correctly downloaded you can run:
```
python -m lidar_gym.tools.map_parser -t
```
To improve training speed and serialize the dataset run:
```
python -m lidar_gym.tools.map_parser -s
```


## Action space
We define action space as a following dictionary:<br />
`action_space = {"rays", "map"}`
where <br />`rays` is 2D binary (numpy.ndarray) matrix representing directions of lidar beams. <br />
`map` is 3D (numpy.ndarray) matrix of map reconstructed by agent.
Environment must receive only local cutout of global map. 
Actually it needs only cuboid of local coordinates `[-16:48, -32:32, -3.2:3.2]` with respect to sensor position in meters. 

## Observation space
We define observation space for `lidar-v1` as following dictionary:<br />
`observation = {"T", "points", "values"}`
where <br />
`T` is 3D (numpy.ndarray) array of transformation matrices to the next positions of the sensor.
Its size is `(N, 4, 4)`.<br />
`points` is 2D (numpy.ndarray) matrix. It is made by points (in rows) found by lidar rays.
Size of this matrix is `(N, 3)`, where N is number of found points. It is None when no points was hit.<br />
`values` is 1D (numpy.ndarray) array. It consist of values corresponding to the
occupancy of points with same index. Its shape is `(1, N)` or None. For each value applies:
```
value < 0 - empty voxel
value == 0 - unknown occupancy
value > 0 - occupied voxel
```
For rest of the environments is observation space ONLY a 3D numpy array with occupancies filled by sparse measurements. Sparse measurements directions are given by actions.
If you want to do supervised learning on dataset, the ground truth map is available in `info`.

## Rendering
Environment offers visualisation for debugging. Use method `render()`. It is available in mode "human" and "ASCII". In following picture is an example of the visualisation of the reconstructed map.

![Visualisation](https://raw.githubusercontent.com/Zdeeno/Bachelor-thesis/master/fig/reconstructed.png)

## Notes
There is a lot of environments available. That's documented in [lidar_gym](lidar_gym/envs/lidar_gym.py) file.
Reward is in range `(-1, 0)`. See [example file](example.py) with initialisation and random action. Currently there
are environments with following parameters:
##### LARGE:
```
fov = (120, 90)
ray density = (160, 120)
voxel size = 0.2
action map size in voxels = (320, 320, 32)
maximum number of rays = 200
lidar range = 48
```

##### SMALL:
```
fov = (120, 90)
ray density = (120, 90)
voxel size = 0.4
action map size in voxels = (160, 160, 16)
maximum number of rays = 50
lidar range = 48
```

##### TOY:
```
fov = (120, 90)
ray density = (40, 30)
voxel size = 0.8
action map size in voxels = (80, 80, 8)
maximum number of rays = 15
lidar range = 48
```

This repository was created for the purpose of this [bachelor thesis](https://github.com/Zdeeno/Bachelor-thesis).