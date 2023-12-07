import pandas as pd
import csv
import matplotlib.pyplot as plt
import numpy as np

from pose_deviations import *

max_avg_ct = 0
max_avg_hd = 0
max_wt_ct  = 0
max_wt_hd  = 0
filename_avg_ct = ''
filename_avg_hd = ''
filename_wt_ct  = ''
filename_wt_hd  = ''

def row_value(target_number, filename, column):

	df = pd.read_csv(filename)

	index_of_number = df.index[df[column] == target_number].tolist()

	return df.iloc[index_of_number]

def worst_case(a , b):
	a = abs(a)
	b = abs(b)

	if abs(a) > abs(b):
		return a
	return b

def analyzer(d, vel, crosstrack, heading):

	global max_avg_ct, max_avg_hd, max_wt_ct, max_wt_hd
	global filename_avg_ct, filename_avg_hd, filename_wt_ct, filename_wt_hd

	filename = "Spiral_5" + "_" + vel + "_" + d + "_" + crosstrack + "_" + heading + ".csv"

	df = pd.read_csv(filename)

	crosstrack_err_per = df['Crosstrack_error_camera']

	crosstrack_err_gaz = df['Crosstrack_error_gt']

	heading_err_per = df['Heading_error_camera']

	heading_err_gaz = df['Heading_error_gt']

	ct_acc = df['Crosstrack_error_accuracy']

	hd_acc = df['heading_accuracy']

	time = df['t']

	# mean and worst groundtruth crosstrack error values
	avg_ct = crosstrack_err_gaz.mean()
	wt_ct = worst_case(crosstrack_err_gaz.min(), crosstrack_err_gaz.max())
	
	# # for plot
	# avg_ct_arr = np.zeros(crosstrack_err_gaz.shape)
	# avg_ct_arr[avg_ct_arr == 0] = avg_ct

	# mean and worst groundtruth heading error values
	avg_hd = heading_err_gaz.mean()
	wt_hd = worst_case(heading_err_gaz.min(), heading_err_gaz.max())
	
	# # for plot
	# avg_hd_arr = np.zeros(heading_err_gaz.shape)
	# avg_hd_arr[avg_hd_arr == 0] = avg_hd

	# # mean and worst crosstrack error accuracy values
	# avg_ct_acc = ct_acc.mean()
	# wt_ct_acc = worst_case(ct_acc.min(), ct_acc.max())

	# # mean and worst heading error accuracy values
	# avg_hd_acc = hd_acc.mean()
	# wt_hd_acc = worst_case(hd_acc.min(), hd_acc.max())

	if max_avg_ct < avg_ct:
		max_avg_ct = avg_ct
		filename_avg_ct = filename

	if max_avg_hd < avg_hd:
		max_avg_hd = avg_hd
		filename_avg_hd = filename
	
	if max_wt_ct < wt_ct:
		max_wt_ct  = wt_ct
		filename_wt_ct = filename
	
	if max_wt_hd < wt_hd:
		max_wt_hd  = wt_hd
		filename_wt_hd = filename

def plotter(filename, plot_param, case):

	df = pd.read_csv(filename)

	crosstrack_err_per = df['Crosstrack_error_camera']

	crosstrack_err_gaz = df['Crosstrack_error_gt']

	heading_err_per = df['Heading_error_camera']

	heading_err_gaz = df['Heading_error_gt']

	ct_acc = df['Crosstrack_error_accuracy']

	hd_acc = df['heading_accuracy']

	time = df['t']

	filename = filename.rsplit(".", 1)[0]
	filename = filename + "_" + case + ".png"

	plot_param_arr = np.zeros(crosstrack_err_gaz.shape)
	plot_param_arr[plot_param_arr == 0] = plot_param

	# Plotting both the curves simultaneously
	plt.plot(time, crosstrack_err_per, color='r', label='Perception Crosstrack Error')
	plt.plot(time, crosstrack_err_gaz, color='g', label='Gazebo Crosstrack Error')
	plt.plot(time, plot_param_arr, color='b', label='mean_gt_Error')

	# Naming the x-axis, y-axis and the whole graph
	plt.xlabel("time")
	plt.ylabel("Error")
	plt.title("Crosstrack_Error")

	# Adding legend, which helps us recognize the curve according to it's color
	plt.legend()

	# To load the display window
	plt.savefig(filename)

	avg_hd = heading_err_gaz.mean()

	avg_hd_arr = np.zeros(heading_err_gaz.shape)
	avg_hd_arr[avg_hd_arr == 0] = avg_hd

	# Adding legend, which helps us recognize the curve according to it's color
	plt.plot(time, heading_err_per, color='r', label='Perception Heading Error')
	plt.plot(time, heading_err_gaz, color='g', label='Gazebo Heading Error')
	plt.plot(time, avg_hd_arr, color='b', label='mean_gt_Error')

	# Naming the x-axis, y-axis and the whole graph
	plt.xlabel("time")
	plt.ylabel("Error")
	plt.title("Heading_Error")

	# Adding legend, which helps us recognize the curve according to it's color
	plt.legend()

	# To load the display window
	plt.savefig(filename)

def main():
	
	pose = deviations()
	diff = [175, 200]
	velocity = [6,9,12]

	for d in [200]:
		for vel in [6]:
			for pos in pose:
				analyzer(str(d), str(vel), str(pos[0]), str(pos[1]))
	
	print("Column values for maximum mean crosstrack error is in " + filename_avg_ct)
	plotter(filename_avg_ct, max_avg_ct, "avg_ct")
	print("\n")

	print("Column values for maximum mean heading error is in " + filename_avg_hd)
	plotter(filename_avg_hd,max_avg_hd, "avg_hd")
	print("\n")

	print("Column values for maximum worst crosstrack error is in " + filename_wt_ct + " and the values are:")
	print(row_value(max_wt_ct, filename_wt_ct, 'Crosstrack_error_gt'))
	plotter(filename_wt_ct, max_wt_ct, "wt_ct")
	print("\n")

	print("Column values for maximum mean heading error is in " + filename_wt_hd + " and the values are:")
	print(row_value(max_wt_hd, filename_wt_hd, 'Heading_error_gt'))
	plotter(filename_wt_hd, max_wt_hd, "wt_hd")

if __name__ == '__main__':
	main()