# Lidar gym
## Dependencies
Environment is written in python 3 using following libraries:

##### 1. Numpy
##### 2. OpenAI gym
https://github.com/openai/gym
##### 3. Pykitti
https://github.com/utiasSTARS/pykitti
##### 4. Voxel map
https://bitbucket.org/tpetricek/voxel_map

Dependencies can by installed by command:<br />`python setup.py install`

## Action space
We define action space as a following tuple:<br />
`action_space = (rays, map)`
where <br />`rays` is 2D binary (numpy) matrix representing directions of lidar beams. <br />
`map` is 3D binary (numpy) matrix of map reconstructed by agent.
Environment must receive only local cutout of global map. 
Actually it needs only cuboid with local coordinates `[-40:40, -20:60, -2:2]`.
Size of your input map should be `(81, 81, 5) ./ voxel_size`. 

## Observation space
We define observation space as following tuple:<br />
`observation = (T, points)`
where <br />
`T` is 2D (numpy) transformation matrix to the next position of sensor.
Its size is `(4, 4)`.<br />
`points` is 2D (numpy) matrix. It is made by points found by lidar rays.
Size of this matrix is `(N, 3)`, where N is number of found points.

## Notes
Currently there is a lot of parameters available. That's documented in [lidar_gym](lidar_gym/lidar_gym.py) file.
Reward is in range `(-inf, 0)`. Package is still under development - gym registration will be added soon.