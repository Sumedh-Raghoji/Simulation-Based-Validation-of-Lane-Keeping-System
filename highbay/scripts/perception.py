#!/usr/bin/env python3

import cv2
import time
import math
import copy
import rospy
import numpy as np
import csv
import matplotlib.pyplot as plt

from skimage import morphology
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Header
from sensor_msgs.msg import Image
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2
from gazebo_msgs.srv import GetModelState, GetModelStateResponse
from tf.transformations import euler_from_quaternion

from Line import Line
from line_fit import line_fit, bird_fit, final_viz

class lanenet_detector:
	def __init__(self):

		self.bridge = CvBridge()
		self.left_line = Line(n=5)
		self.right_line = Line(n=5)
		self.detected = False


		# initialization for the lateral tracking error and lane heading
		self.lateral_error = 0.0
		self.lane_theta = 0.0
		self.ref_theta = 0.0
		self.average_curverad = 0.0
		self.ploty = 0.0
		self.left_fitx = 0.0
		self.right_fitx = 0.0
		self.center_fitx = 0.0
		# determine the meter-to-pixel ratio
		lane_width_meters = 4.4
		lane_width_pixels = 265.634
		self.meter_per_pixel = lane_width_meters / lane_width_pixels

		# Subscriber must be declared after intializing other attributes
		self.pub_bird = rospy.Publisher("lane_detection/birdseye", Image, queue_size=1)
		self.warp = rospy.Publisher("lane_detection/warp", Image, queue_size=1)

	def img_callback(self, data):

		try:
			# Convert a ROS image message into an OpenCV image
			cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
		except CvBridgeError as e:
			print(e)

		raw_img = cv_image.copy()
		mask_image, bird_image, lateral_error, lane_theta, warp_img, average_curverad, ploty, left_fitx, right_fitx, center_fitx = self.detection(raw_img)

		if mask_image is not None and bird_image is not None:

			# Convert an OpenCV image into a ROS image message

			# out_img_msg = self.bridge.cv2_to_imgmsg(mask_image, 'bgr8')
			out_bird_msg = self.bridge.cv2_to_imgmsg(bird_image, 'bgr8')
			warp_img = self.bridge.cv2_to_imgmsg(warp_img, 'bgr8')

			# Publish image message in ROS
			# self.pub_image.publish(out_img_msg)
			self.pub_bird.publish(out_bird_msg)
			self.warp.publish(warp_img)

		# publish the lateral tracking error and lane heading
		self.pub_image = mask_image
		if lateral_error is not None:
			self.lateral_error = lateral_error

		if lane_theta is not None:
			self.lane_theta = lane_theta
		
		if average_curverad is not None:
			self.average_curverad = average_curverad
		
		if ploty is not None:
			self.ploty = ploty
		
		if left_fitx is not None:
			self.left_fitx = left_fitx
		
		if right_fitx is not None:
			self.right_fitx = right_fitx
		
		if center_fitx is not None:
			self.center_fitx = center_fitx

		return self.pub_image, self.lateral_error, self.lane_theta, self.average_curverad


	def gradient_thresh(self, img, thresh_min=25, thresh_max=100):

		#Applying sobel edge detection

		gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

		blur_image = cv2.GaussianBlur(gray_image, (3, 3), cv2.BORDER_DEFAULT)

		sobelx = cv2.Sobel(blur_image, cv2.CV_8U,1,0,ksize=3)
		sobely = cv2.Sobel(blur_image, cv2.CV_8U,0,1,ksize=3)

		# Step 4
		combined_img = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0.0) # alpha = beta = 0.5, gamma = 0.0

		# Step 5
		binary_img = np.zeros_like(combined_img)
		threshold = (combined_img > thresh_min) & (combined_img < thresh_max)
		# binary_img[threshold] = 1
		binary_img[threshold] = 255

		return binary_img
	

	def color_thresh(self, img, thresh=(100, 255)):

		#RGB to HLS
		
		HLS_image = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

		x, y, channel  = HLS_image.shape
		binary_img = np.zeros((x, y))
		H, L, S = HLS_image[:, :, 0], HLS_image[:, :, 1], HLS_image[:, :, 2]
		s_threshold = ( S > thresh[0]) & ( S < thresh[1])

		l_threshold = L > 150
		h_threshold = ( H < 40 ) | (H > 150) # FILTERING OUT GREEN
		# binary_img[(s_threshold | l_threshold) & h_threshold] =  1
		binary_img[(s_threshold | l_threshold) & h_threshold] = 255

		return binary_img
	

	def combinedBinaryImage(self, img):
		
		ColorOutput = self.color_thresh(img)
		# SobelOutput = self.gradient_thresh(img)
		####
 
		# binaryImage = np.zeros_like(SobelOutput)
		# binaryImage[(ColorOutput==1)|(SobelOutput==1)] = 1
		# binaryImage[(ColorOutput==1)|(SobelOutput==1)] = 255

		# Remove noise from binary image
		# binaryImage = morphology.remove_small_objects(binaryImage.astype('bool'),min_size=50,connectivity=2)
	
		return ColorOutput

	def perspective_transform(self, img, verbose=False):

		# bird's eye view
		# Image = (480, 640)

		# a = np.float32([[x1, y1], [x2,y2], [x3, y3], [x4, y4]])
		input = np.float32([[240, 280], [420,280], [620, 430], [20, 430]])
		output = np.float32([[255, 0], [435, 0], [450, 600], [215, 600]])

		M = cv2.getPerspectiveTransform(input, output)
		Minv = np.linalg.inv(M)

		new_img = np.uint8(img)
		warped_img = cv2.warpPerspective(new_img, M, (img.shape[1], img.shape[0]))

		return warped_img, M, Minv
	
	def pi_2_pi(self, angle):
		if angle > math.pi:
			return angle - 2.0 * math.pi
		if angle < -math.pi:
			return angle + 2.0 * math.pi

		return angle

	def detection(self, img, prev=time.time()):

		binary_img = self.combinedBinaryImage(img)

		img_birdeye, M, Minv = self.perspective_transform(binary_img)

		curr_pos_pixel_x = img_birdeye.shape[1]/2
		curr_pos_pixel_y = 850

		# Fit lane with the newest image
		ret, average_curverad, ploty, left_fitx, right_fitx, center_fitx = line_fit(img_birdeye)
		
		if ret is not None:
			left_fit = ret['left_fit']
			right_fit = ret['right_fit']
			nonzerox = ret['nonzerox']
			nonzeroy = ret['nonzeroy']
			left_lane_inds = ret['left_lane_inds']
			right_lane_inds = ret['right_lane_inds']
			warp_img = ret['out_img']
			center_fit = ret['center_fit']

			left_fit = self.left_line.add_fit(left_fit)
			right_fit = self.right_line.add_fit(right_fit)

		# return lane detection results
		bird_fit_img = None
		combine_fit_img = None
		lateral_error = None
		lane_theta = None
		if ret is not None:
			bird_fit_img = bird_fit(img_birdeye, ret, save_file=None)

			combine_fit_img = final_viz(img, left_fit, right_fit, Minv)

			# Error calculations
			x = center_fit[2] #y=0
			xintercept = center_fit[0]* curr_pos_pixel_y**2 + center_fit[1]* curr_pos_pixel_y + center_fit[2] #curr_pos_pixel_y = 850

			lane_theta = math.atan2((x - xintercept), (curr_pos_pixel_y - 0))
			lane_theta = - self.pi_2_pi(lane_theta)

			lateral_pixel_error = (curr_pos_pixel_x - xintercept) * math.cos(lane_theta)
			lateral_error = lateral_pixel_error*self.meter_per_pixel

		else:
			print("Unable to detect lanes")

		return combine_fit_img, bird_fit_img, lateral_error, lane_theta, warp_img, average_curverad, ploty, left_fitx, right_fitx, center_fitx
