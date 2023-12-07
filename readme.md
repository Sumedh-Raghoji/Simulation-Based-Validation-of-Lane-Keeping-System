Vision-based Lane Following:

This repository hosts the codebase for simulation of Polaris_GEM_e2 vehicle in Gazebo. Lane following goal is achieved using only monochrome camera on different tracks. Stanley controller is used to achieve the lateral control. This repository can be used as base for advanced topic such as camera based localization, longitudinal controller design, controller performance testing, etc.

Requirements:

$ sudo apt-get install ros-noetic-effort-controllers  
$ sudo apt install ros-noetic-ackermann-msgs  
$ pip3 install numpy  
$ pip3 install matplotlib  

Running the Simulation:

# To launch the simulator
$ source devel/setup.bash  
$ roslaunch highbay highbay.launch  

# To run the simulator
$ source devel/setup.bash  
$ rosrun highbay gem_vision.py  

# TO return the vehicle back to initial position
$ source devel/setup.bash  
$ python3 reset.py  
