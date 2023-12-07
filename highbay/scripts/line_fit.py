import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle

# feel free to adjust the parameters in the code if necessary

def line_fit(binary_warped):
	"""
	Find and fit lane lines
	"""
	# Assuming you have created a warped binary image called "binary_warped"
	# Take a histogram of the bottom half of the image
	histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
	
	# Create an output image to draw on and visualize the result
	out_img = (np.dstack((binary_warped, binary_warped, binary_warped))*255).astype('uint8')

	# Find the peak of the left and right halves of the histogram
	# These will be the starting point for the left and right lines
	midpoint = int(histogram.shape[0]/2)
	# leftx_base = np.argmax(histogram[100:midpoint]) + 100
	leftx_base = np.argmax(histogram[:midpoint])
	rightx_base = np.argmax(histogram[midpoint:-100]) + midpoint

	# Choose the number of sliding windows
	nwindows = 9
	# Set height of windows
	window_height = int(binary_warped.shape[0]/nwindows)
	# Identify the x and y positions of all nonzero pixels in the image
	nonzero = binary_warped.nonzero()
	nonzeroy = np.array(nonzero[0])
	nonzerox = np.array(nonzero[1])
	# Current positions to be updated for each window
	leftx_current = leftx_base
	rightx_current = rightx_base
	# Set the width of the windows +/- margin
	margin = 80
	# margin = 50

	# Set minimum number of pixels found to recenter window
	minpix = 50
	# Create empty lists to receive left and right lane pixel indices
	left_lane_inds = []
	right_lane_inds = []

	# Step through the windows one by one
	for window in range(nwindows):
		# Identify window boundaries in x and y (and right and left)
		##TO DO
		window_bottom = binary_warped.shape[0] - (window)*window_height
		window_top = binary_warped.shape[0] - (window+1)*window_height
		left_window_x0 = leftx_current - margin
		left_window_x1 = leftx_current + margin
		right_window_x0 = rightx_current - margin
		right_window_x1 = rightx_current + margin	

		# left window start/end points
		
		# 1. Reverse order of Asher/Arjun's code
		# 2. From cv2 doc: (x, y) is not (row, col)
		start_left = (left_window_x0, window_top)
		end_left =   (left_window_x1, window_bottom)
		
		# right window start/end points
		start_right = (right_window_x0, window_top)
		end_right = (right_window_x1, window_bottom)

		# draw left and right window
		green = (0,255,0) # color in BGR
		thickness = 3

		out_img = cv2.rectangle(out_img,start_left, end_left, green, thickness)
		out_img = cv2.rectangle(out_img,start_right, end_right, green, thickness)

		####
		# Identify the nonzero pixels in x and y within the window
		##TO DO
		
		# Mask out out-of-range pixels so we don't need to worry about index
		mask_top = nonzeroy >= window_top
		mask_bottom = nonzeroy <= window_bottom
		mask_left_x0 = nonzerox >= left_window_x0
		mask_left_x1 = nonzerox <= left_window_x1
		mask_right_x0 = nonzerox >= right_window_x0
		mask_right_x1 = nonzerox <= right_window_x1

		nonzero_left_idx = np.nonzero((mask_top & mask_bottom & mask_left_x0 & mask_left_x1))[0] # due to tuple
		nonzero_right_idx = np.nonzero((mask_top & mask_bottom & mask_right_x0 & mask_right_x1))[0] # due to tuple
		
		# Append these indices to the lists
		
		left_lane_inds.append(nonzero_left_idx)
		right_lane_inds.append(nonzero_right_idx)

		# If found > minpix pixels, recenter next window on their mean position

		if len(nonzero_left_idx) > minpix:
			leftx_current = int(np.mean(nonzerox[nonzero_left_idx]))
		if len(nonzero_right_idx) > minpix:
			rightx_current = int(np.mean(nonzerox[nonzero_right_idx]))
		
	# Concatenate the arrays of indices
	left_lane_inds = np.concatenate(left_lane_inds)
	right_lane_inds = np.concatenate(right_lane_inds)

	# Extract left and right line pixel positions
	leftx = nonzerox[left_lane_inds]
	lefty = nonzeroy[left_lane_inds]
	rightx = nonzerox[right_lane_inds]
	righty = nonzeroy[right_lane_inds]

	left_lane_undetected  = 0
	right_lane_undetected = 0
	
	if (len(leftx) != 0):
		left_fit = np.polyfit(lefty, leftx, 2)
	else:
		left_lane_undetected = 1

	if (len(rightx) != 0):
		right_fit = np.polyfit(righty, rightx, 2)
	else:
		right_lane_undetected = 1


	# Generate x and y values for plotting
	ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])

	if (left_lane_undetected == 0):
		left_fitx  = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
	else:
		left_fitx  = 1*ploty**2 + 1*ploty

	if (right_lane_undetected == 0):
		right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
	else:
		right_fitx = 1*ploty**2 + 1*ploty

	# Colors in the left and right lane regions
	out_img[lefty, leftx]   = [255, 0, 0]
	out_img[righty, rightx] = [0, 0, 255]

	left_pts = np.array(list(zip(left_fitx,ploty)), np.int32)
	left_pts = left_pts.reshape((-1, 1, 2))

	right_pts = np.array(list(zip(right_fitx,ploty)), np.int32)
	right_pts = right_pts.reshape((-1, 1, 2))


	if (left_lane_undetected == 0):
		cv2.polylines(out_img, [left_pts], False, (0,255,255), 2)

	if (right_lane_undetected == 0):    
		cv2.polylines(out_img, [right_pts], False, (0,255,255), 2)

	y_eval = np.max(ploty)

	if (left_lane_undetected == 0):
		left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
	else:
		left_curverad = 400000


	if (right_lane_undetected == 0):
		right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
	else:
		right_curverad = 400000


	if (left_lane_undetected == 0) and (left_curverad > 100000):
		center_fitx = left_fitx + 120
	elif (right_lane_undetected == 0) and (right_curverad > 100000):
		center_fitx = right_fitx - 120
	elif (left_lane_undetected == 0) and (right_lane_undetected == 0):
		center_fitx = (left_fitx + right_fitx)/2
	
	center_fit = [(left_fit[0]+ right_fit[0])/2, (left_fit[1]+ right_fit[1])/2, (left_fit[2]+ right_fit[2])/2]

	center_fitx  = center_fit[0]*ploty**2 + center_fit[1]*ploty + center_fit[2]

	center_pts = np.array(list(zip(center_fitx,ploty)), np.int32)
	center_pts = center_pts.reshape((-1, 1, 2))

	curr_pos_pixel_x = out_img.shape[1]/2
	curr_pos_pixel_y = out_img.shape[0]

	x_1 = center_fit[2]
	xintercept = center_fit[0]* curr_pos_pixel_y**2 + center_fit[1]* curr_pos_pixel_y + center_fit[2]

	x = [x_1, xintercept]
	y = [0, curr_pos_pixel_y]

	center_pts = np.array(list(zip(x,y)), np.int32)
	center_pts = center_pts.reshape((-1, 1, 2))

	cv2.polylines(out_img, [center_pts], False, (255,255,255), 3)

	fwd_dir_pts = np.array(list(zip([320,320],[0,480])), np.int32)
	fwd_dir_pts = fwd_dir_pts.reshape((-1, 1, 2))

	cv2.polylines(out_img, [fwd_dir_pts], False, (120,120,120), 3)

	average_curverad = (left_curverad + right_curverad)/2.0 
	average_curverad = average_curverad/5.0

	# Return a dict of relevant variables
	ret = {}
	ret['left_fit'] = left_fit
	ret['right_fit'] = right_fit
	ret['center_fit'] = center_fit
	ret['nonzerox'] = nonzerox
	ret['nonzeroy'] = nonzeroy
	ret['out_img'] = out_img
	ret['left_lane_inds'] = left_lane_inds
	ret['right_lane_inds'] = right_lane_inds

	# return ret
	return ret, average_curverad/100, ploty, left_fitx, right_fitx, center_fitx

def bird_fit(binary_warped, ret, save_file=None):
	"""
	Visualize the predicted lane lines with margin, on binary warped image
	save_file is a string representing where to save the image (if None, then just display)
	"""
	# Grab variables from ret dictionary
	left_fit = ret['left_fit']
	right_fit = ret['right_fit']
	nonzerox = ret['nonzerox']
	nonzeroy = ret['nonzeroy']
	left_lane_inds = ret['left_lane_inds']
	right_lane_inds = ret['right_lane_inds']

	# Create an image to draw on and an image to show the selection window
	out_img = (np.dstack((binary_warped, binary_warped, binary_warped))*255).astype('uint8')
	window_img = np.zeros_like(out_img)
	# Color in left and right line pixels
	out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
	out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

	# Generate x and y values for plotting
	ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
	left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
	right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

	# Generate a polygon to illustrate the search window area
	# And recast the x and y points into usable format for cv2.fillPoly()
	margin = 100  # NOTE: Keep this in sync with *_fit()
	left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
	left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
	left_line_pts = np.hstack((left_line_window1, left_line_window2))
	right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
	right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
	right_line_pts = np.hstack((right_line_window1, right_line_window2))

	# Draw the lane onto the warped blank image
	cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
	cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
	result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

	plt.imshow(result)
	plt.plot(left_fitx, ploty, color='yellow')
	plt.plot(right_fitx, ploty, color='yellow')
	plt.xlim(0, 1280)
	plt.ylim(720, 0)

	# cv2.imshow('bird',result)
	# cv2.imwrite('bird_from_cv2.png', result)

	# if save_file is None:
	# 	plt.show()
	# else:
	# 	plt.savefig(save_file)
	# plt.gcf().clear()

	return result

def final_viz(undist, left_fit, right_fit, m_inv):
	"""
	Final lane line prediction visualized and overlayed on top of original image
	"""
	# Generate x and y values for plotting
	ploty = np.linspace(0, undist.shape[0]-1, undist.shape[0])
	left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
	right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
	
	# Create an image to draw the lines on
	#warp_zero = np.zeros_like(warped).astype(np.uint8)
	#color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
	color_warp = np.zeros((720, 1280, 3), dtype='uint8')  # NOTE: Hard-coded image dimensions

	# Recast the x and y points into usable format for cv2.fillPoly()
	pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
	pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
	pts = np.hstack((pts_left, pts_right))

	# Draw the lane onto the warped blank image
	cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

	# Warp the blank back to original image space using inverse perspective matrix (Minv)
	newwarp = cv2.warpPerspective(color_warp, m_inv, (undist.shape[1], undist.shape[0]))

	# Combine the result with the original image
	# Convert arrays to 8 bit for later cv to ros image transfer
	undist = np.array(undist, dtype=np.uint8)
	newwarp = np.array(newwarp, dtype=np.uint8)

	result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)

	return result