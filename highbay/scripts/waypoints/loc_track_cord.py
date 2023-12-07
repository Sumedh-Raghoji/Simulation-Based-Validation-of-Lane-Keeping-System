import matplotlib.pyplot as plt
import numpy as np
import csv

x = []
y = []

z = 0.1
s  = "<point>"
s1 = "</point>"
filename = "loc_track_cord.csv"
c = str(z)

r = 50

for i in range(-80, 90):
    i = np.radians(i)
    x1 = r*np.cos(i)
    y1 = r*np.sin(i) + 50
    x.append(x1)
    y.append(y1)
    x1 = str(x1)
    y1 = str(y1)
    row = s + " " + x1 + " " +  y1 + " " + c + " " + s1
    rows = [[row]]
    with open(filename, 'a') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(rows)

r = 30

a = 50*np.cos(np.radians(90))
b = 50*np.sin(np.radians(90)) + 50 - 30
for i in range(90, 180):
    i = np.radians(i)
    x1 = a + r*np.cos(i)
    y1 = b + r*np.sin(i) 
    x.append(x1)
    y.append(y1)
    x1 = str(x1)
    y1 = str(y1)
    row = s + " " + x1 + " " +  y1 + " " + c + " " + s1
    rows = [[row]]
    with open(filename, 'a') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(rows)

r = 40

a = a + 30*np.cos((np.radians(180))) + r
b = b + 30*np.sin((np.radians(180))) 
for i in range(180, 210):
    i = np.radians(i)
    x1 = a + r*np.cos(i)
    y1 = b + r*np.sin(i)
    x.append(x1)
    y.append(y1)
    x1 = str(x1)
    y1 = str(y1)
    row = s + " " + x1 + " " +  y1 + " " + c + " " + s1
    rows = [[row]]
    with open(filename, 'a') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(rows)

# r = 20

# a = a + 40*np.cos(np.radians(230))
# b = b + 40*np.sin(np.radians(230)) - r
# for i in range(90, 30, -1):
#     i = np.radians(i)
#     x1 = a + r*np.cos(i)
#     y1 = b + r*np.sin(i)
#     x.append(x1)
#     y.append(y1)
    # x1 = str(x1)
    # y1 = str(y1)
    # row = s + " " + x1 + " " +  y1 + " " + c + " " + s1
    # rows = [[row]]
    # with open(filename, 'a') as csvfile:
    #     csvwriter = csv.writer(csvfile)
    #     csvwriter.writerows(rows)

a = a + r*np.cos(np.radians(210))
b = b + r*np.sin(np.radians(210))
e = 50*np.cos(np.radians(-80))
d = 50*np.sin(np.radians(-80)) + 50

for i in ([a,b], [e, d]):
    x1 = i[0]
    y1 = i[1]
    x.append(x1)
    y.append(y1)
    x1 = str(x1)
    y1 = str(y1)
    row = s + " " + x1 + " " +  y1 + " " + c + " " + s1
    rows = [[row]]
    with open(filename, 'a') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(rows)

# plt.plot(x, y, color='b', label='track')

# # Naming the x-axis, y-axis and the whole graph
# plt.xlabel("x")
# plt.ylabel("y")
# plt.title("track")

# # Adding legend, which helps us recognize the curve according to it's color
# plt.legend()
# plt.show()