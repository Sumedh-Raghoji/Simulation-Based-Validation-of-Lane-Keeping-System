#!/usr/bin/env python3

#================================================================
# File name: gem_vision.py                                                                  
# Description: lane detection with frontal camera                                                           
# Author: Hang Cui
# Email: hangcui3@illinois.edu                                                                     
# Date created: 05/20/2021                                                                
# Date last modified: 06/12/2021                                                          
# Version: 0.1                                                                    
# Usage: rosrun gem_vision gem_vision.py                                                                      
# Python version: 3.8                                                             
#================================================================

from __future__ import print_function

import sys
import copy
import time
import rospy
import rospkg
import math

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from std_msgs.msg import String, Float64
from geometry_msgs.msg import Point
from sensor_msgs.msg import Image, CameraInfo
from image_geometry import PinholeCameraModel
from cv_bridge import CvBridge, CvBridgeError
from ackermann_msgs.msg import AckermannDrive

from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.srv import GetModelState
from std_msgs.msg import Float32MultiArray

from perception import lanenet_detector
from stanley_sim import *

class ImageConverter:

	def __init__(self):

		self.node_name = "gem_vision"
		
		rospy.init_node(self.node_name)
		
		rospy.on_shutdown(self.cleanup)

		self.f_steer_prev = 0
		self.f_steer_curr = 0
		self.ackermann_msg = AckermannDrive()
		self.ackermann_msg.steering_angle_velocity = 0.0
		self.ackermann_msg.acceleration            = 0.0
		self.ackermann_msg.jerk                    = 0.0
		self.ackermann_msg.speed                   = 0.0 
		self.ackermann_msg.steering_angle          = 0.0
		self.wheelbase  = 1.75 
		
		# Create the cv_bridge object
		self.bridge          = CvBridge()

		self.frame_width  = 640
		self.frame_height = 480
		self.bob_readings = []

		self.ackermann_pub            = rospy.Publisher('/ackermann_cmd', AckermannDrive, queue_size=0)
		self.image_pub_debug 		  = rospy.Publisher("/front_camera/image_warped", Image, queue_size=1)
		self.image_pub       		  = rospy.Publisher("/front_camera/image_processed", Image, queue_size=1)
		self.curve_rad_pub       	  = rospy.Publisher("/curve_rad", Float64, queue_size=1)
		self.image_sub       		  = rospy.Subscriber("/front_single_camera/image_raw", Image, self.image_callback)
		self.warp_img_sub       	  = rospy.Subscriber("lane_detection/warp", Image, self.image_callback0)
	
	def euler_from_quaternion(self, x, y, z, w):
		"""
		Convert a quaternion into euler angles (roll, pitch, yaw)
		roll is rotation around x in radians (counterclockwise)
		pitch is rotation around y in radians (counterclockwise)
		yaw is rotation around z in radians (counterclockwise)
		"""
		t0 = +2.0 * (w * x + y * z)
		t1 = +1.0 - 2.0 * (x * x + y * y)
		roll_x = math.atan2(t0, t1)
	 
		t2 = +2.0 * (w * y - z * x)
		t2 = +1.0 if t2 > +1.0 else t2
		t2 = -1.0 if t2 < -1.0 else t2
		pitch_y = math.asin(t2)
	 
		t3 = +2.0 * (w * z + x * y)
		t4 = +1.0 - 2.0 * (y * y + z * z)
		yaw_z = math.atan2(t3, t4)
	 
		return roll_x, pitch_y, yaw_z # in radians

	def robot_readings(self):
		return self.bob_readings

	def image_callback0(self, image):
		self.warp_img = self.bridge.imgmsg_to_cv2(image, "bgr8")

	def image_callback(self, ros_image, prev=time.time()):

		perceptionModule = lanenet_detector()

		pub_image, lateral_error_camera, lane_theta_camera, center_curverad = perceptionModule.img_callback(ros_image)

		self.curve_rad_pub.publish(center_curverad)

		# Errors calculated by Camera
		ef =  lateral_error_camera
		theta_e = lane_theta_camera

		# Velocity
		f_vel = 3

		# Calculating the steering angle using stanley controller
		delta = round(theta_e + math.atan2(ef, f_vel), 3)

		# Steering angle and velocity commands
		self.ackermann_msg.speed          = f_vel
		self.ackermann_msg.steering_angle = delta
		self.ackermann_pub.publish(self.ackermann_msg)
		
		# Use cv_bridge() to convert the ROS image to OpenCV format
		try:
			frame = self.bridge.imgmsg_to_cv2(ros_image, "bgr8")
		except CvBridgeError as e:
			rospy.logerr("CvBridge Error: {0}".format(e))

		if (center_curverad > 200):
			curve_txt = "Lane: straight"
		else:
			curve_txt = "Lane: curved"

		font = cv2.FONT_HERSHEY_DUPLEX
		cv2.putText(pub_image, curve_txt, (20, 30), font, 1, (255,0,0), 2)
		cv2.putText(pub_image,'%.3f m' % (center_curverad), (60, 130), font, 2, (0,255,0), 2, cv2.LINE_AA)

		try:
			# Convert OpenCV image to ROS image and publish
			self.image_pub.publish(self.bridge.cv2_to_imgmsg(pub_image, "bgr8"))
			self.image_pub_debug.publish(self.bridge.cv2_to_imgmsg(self.warp_img, "bgr8"))
		except CvBridgeError as e:
			rospy.logerr("CvBridge Error: {0}".format(e))

	def cleanup(self):
		print ("Shutting down vision node.")
		cv2.destroyAllWindows()
	

def main():

	try:
		ImageConverter()
		rospy.spin()
	except KeyboardInterrupt:
		print ("Shutting down vision node.")
		cv2.destroyAllWindows()

if __name__ == '__main__':
	main()