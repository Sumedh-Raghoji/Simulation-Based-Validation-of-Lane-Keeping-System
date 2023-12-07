import matplotlib.pyplot as plt
import numpy as np
import csv


x = []
y = []

z = 0.1
s  = "<point>"
s1 = "</point>"
filename = "spiral_10_waypoints.csv"
c = str(z)

r = 50
a = 2.5

for i in range(0, 90):
    i = np.radians(i)
    x1 = (r + a)*np.cos(i)
    y1 = (r + a)*np.sin(i)
    x.append(x1)
    y.append(y1)
    # x1 = str(x1)
    # y1 = str(y1)
    # row = s + " " + x1 + " " +  y1 + " " + c + " " + s1
    row = [x1, y1, z]
    rows = [row]
    with open(filename, 'a') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(rows)


r = 40

for i in range(90, 180):
    i = np.radians(i)
    x1 = (r + a)*np.cos(i)
    y1 = (r + a)*np.sin(i) + 10
    x.append(x1)
    y.append(y1)
    # x1 = str(x1)
    # y1 = str(y1)
    # row = s + " " + x1 + " " +  y1 + " " + c + " " + s1
    # row = x1 + "," +  y1 + "," + c
    # rows = [[row]]
    row = [x1, y1, z]
    rows = [row]
    with open(filename, 'a') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(rows)

r = 30

for i in range(180, 270):
    i = np.radians(i)
    x1 = (r + a)*np.cos(i) - 10
    y1 = (r + a)*np.sin(i) + 10
    x.append(x1)
    y.append(y1)
    # x1 = str(x1)
    # y1 = str(y1)
    # row = s + " " + x1 + " " +  y1 + " " + c + " " + s1
    # row = x1 + "," +  y1 + "," + c
    # rows = [[row]]
    row = [x1, y1, z]
    rows = [row]
    with open(filename, 'a') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(rows)

r = 20

for i in range(270, 360):
    i = np.radians(i)
    x1 = (r + a)*np.cos(i) - 10
    y1 = (r + a)*np.sin(i)
    x.append(x1)
    y.append(y1)
    # x1 = str(x1)
    # y1 = str(y1)
    # row = s + " " + x1 + " " +  y1 + " " + c + " " + s1
    # row = x1 + "," +  y1 + "," + c
    # rows = [[row]]
    row = [x1, y1, z]
    rows = [row]
    with open(filename, 'a') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(rows)

r = 10

for i in range(0, 90):
    i = np.radians(i)
    x1 = (r + a)*np.cos(i)
    y1 = (r + a)*np.sin(i)
    x.append(x1)
    y.append(y1)
    # x1 = str(x1)
    # y1 = str(y1)
    # row = s + " " + x1 + " " +  y1 + " " + c + " " + s1
    # row = x1 + "," +  y1 + "," + c
    # rows = [[row]]
    row = [x1, y1, z]
    rows = [row]
    with open(filename, 'a') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(rows)

plt.plot(x, y, color='b', label='track')

# Naming the x-axis, y-axis and the whole graph
plt.xlabel("x")
plt.ylabel("y")
plt.title("track")

# Adding legend, which helps us recognize the curve according to it's color
plt.legend()
plt.show()