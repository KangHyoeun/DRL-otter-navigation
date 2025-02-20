**DRL Robot navigation in IR-SIM**

Deep Reinforcement Learning algorithm implementation for simulated robot navigation in IR-SIM. Using 2D laser sensor data
and information about the goal point a robot learns to navigate to a specified point in the environment.

![Example](https://github.com/reiniscimurs/DRL-robot-navigation-IR-SIM/blob/master/image.png)

**Installation**

* Package versioning is managed with poetry \
`pip install poetry`
* Clone the repository \
`git clone https://github.com/reiniscimurs/DRL-robot-navigation.git`
* Navigate to the cloned location and install using poetry \
`poetry install`

**Training the model**

* Run the training by executing the train.py file \
`poetry run python robot_nav/train.py`



**Sources**

| Package/Model |                          Source                           | 
|:--------------|:---------------------------------------------------------:| 
| IR-SIM        |            https://github.com/hanruihua/ir-sim            | 
| TD3           | https://github.com/reiniscimurs/DRL-Robot-Navigation-ROS2 | 
| SAC           |        https://github.com/denisyarats/pytorch_sac         | 
| PPO           |      https://github.com/nikhilbarhate99/PPO-PyTorch       | 
| DDPG          |                     Updated from TD3                      | 


