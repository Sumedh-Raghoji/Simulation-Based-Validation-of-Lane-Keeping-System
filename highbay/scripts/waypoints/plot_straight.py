import matplotlib.pyplot as plt
import numpy as np
import csv

x = []
y = []

fc = 25

z = 0.1
s  = "<point>"
s1 = "</point>"
name = "waypoints_straight.csv"
c = str(z)

for i in range(80, -100, -1):
    a = i
    b = -21.5
    # x.append(a)
    # y.append(b)
    # a = str(a)
    # b = str(b)
    # row = s +" "+a+" "+b+" "+c+" "+ s1
    row = [a, b, z]
    rows = [row]
    filename = name
    with open(filename, 'a') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(rows)

# plt.plot(x,y, color='r', label='Straight_Road')
# plt.xlabel('X-axis')
# plt.ylabel('Y-axis')
# plt.title("Track")
# # plt.legend()
# plt.show()