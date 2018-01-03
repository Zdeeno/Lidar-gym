# Lidar gym
## Dependencies
Environment is written in python 3 using following libraries:

##### 1. Numpy
https://github.com/numpy/numpy
##### 2. OpenAI gym
https://github.com/openai/gym
##### 3. Pykitti
https://github.com/utiasSTARS/pykitti
##### 4. Voxel map
https://bitbucket.org/tpetricek/voxel_map

Dependencies can by installed by command:<br />`python setup.py install`

## Action space
We define action space as a following dictionary:<br />
`action_space = {"rays", "map"}`
where <br />`rays` is 2D binary (numpy.ndarray) matrix representing directions of lidar beams. <br />
`map` is 3D (numpy.ndarray) matrix of map reconstructed by agent.
Environment must receive only local cutout of global map. 
Actually it needs only cuboid with local coordinates `[-20:60, -40:40, -2:2]`.
Size of your input map should be `((80, 80, 4) ./ voxel_size) + (1, 1, 1)`. 

## Observation space
We define observation space as following dictionary:<br />
`observation = {"T", "points", "values"}`
where <br />
`T` is 2D (numpy.ndarray) transformation matrix to the next position of sensor.
Its size is `(4, 4)`.<br />
`points` is 2D (numpy.ndarray) matrix. It is made by points (in rows) found by lidar rays.
Size of this matrix is `(N, 3)`, where N is number of found points. It is None when no points was hit.<br />
`values` is 1D (numpy.ndarray) array. It consist of values corresponding to the
occupancy of points with same index. Its size is `(1, N)` or None. For each value applies:
```
value < 0 - empty voxel
value == 0 - unknown occupancy
value > 0 - occupied voxel
```
## Notes
Currently there is a lot of parameters available. That's documented in [lidar_gym](lidar_gym/envs/lidar_gym.py) file.
Reward is in range `(-inf, 0)`. Package is still under development.