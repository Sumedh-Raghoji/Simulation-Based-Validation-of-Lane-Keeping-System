import matplotlib.pyplot as plt
import numpy as np
import csv


x = []
y = []

z = 0.1
s  = "<point>"
s1 = "</point>"
filename = "spiral_5_inner.csv"
c = str(z)

r = 50
a = -5

for i in range(0, 90):
    i = np.radians(i)
    x1 = (r + a)*np.cos(i)
    y1 = (r + a)*np.sin(i)
    # x.append(x1)
    # y.append(y1)
    x1 = str(x1)
    y1 = str(y1)
    row = s + " " + x1 + " " +  y1 + " " + c + " " + s1
    rows = [[row]]
    with open(filename, 'a') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(rows)


r = 45

for i in range (90, 180):
    i = np.radians(i)
    x1 = (r + a)*np.cos(i)
    y1 = 5 + (r + a)*np.sin(i)
    # x.append(x1)
    # y.append(y1)
    x1 = str(x1)
    y1 = str(y1)
    row = s + " " + x1 + " " +  y1 + " " + c + " " + s1
    rows = [[row]]
    with open(filename, 'a') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(rows)

r = 40

for i in range (180, 270):
    i = np.radians(i)
    x1 = -5 + (r + a)*np.cos(i)
    y1 = 5 + (r + a)*np.sin(i)
    # x.append(x1)
    # y.append(y1)
    x1 = str(x1)
    y1 = str(y1)
    row = s + " " + x1 + " " +  y1 + " " + c + " " + s1
    rows = [[row]]
    with open(filename, 'a') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(rows)

r = 35

for i in range (270, 360):
    i = np.radians(i)
    x1 = -5 + (r + a)*np.cos(i)
    y1 = (r + a)*np.sin(i)
    # x.append(x1)
    # y.append(y1)
    x1 = str(x1)
    y1 = str(y1)
    row = s + " " + x1 + " " +  y1 + " " + c + " " + s1
    rows = [[row]]
    with open(filename, 'a') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(rows)

r = 30

for i in range (0, 90):
    i = np.radians(i)
    x1 = (r + a)*np.cos(i)
    y1 = (r + a)*np.sin(i)
    # x.append(x1)
    # y.append(y1)
    x1 = str(x1)
    y1 = str(y1)
    row = s + " " + x1 + " " +  y1 + " " + c + " " + s1
    rows = [[row]]
    with open(filename, 'a') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(rows)

r = 25

for i in range (90, 180):
    i = np.radians(i)
    x1 = (r + a)*np.cos(i)
    y1 = 5 + (r + a)*np.sin(i)
    # x.append(x1)
    # y.append(y1)
    x1 = str(x1)
    y1 = str(y1)
    row = s + " " + x1 + " " +  y1 + " " + c + " " + s1
    rows = [[row]]
    with open(filename, 'a') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(rows)

r = 20

for i in range (180, 270):
    i = np.radians(i)
    x1 = -5 + (r + a)*np.cos(i)
    y1 = 5 + (r + a)*np.sin(i)
    # x.append(x1)
    # y.append(y1)
    x1 = str(x1)
    y1 = str(y1)
    row = s + " " + x1 + " " +  y1 + " " + c + " " + s1
    rows = [[row]]
    with open(filename, 'a') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(rows)

r = 15

for i in range (270, 360):
    i = np.radians(i)
    x1 = -5 + (r + a)*np.cos(i)
    y1 = (r + a)*np.sin(i)
    # x.append(x1)
    # y.append(y1)
    x1 = str(x1)
    y1 = str(y1)
    row = s + " " + x1 + " " +  y1 + " " + c + " " + s1
    rows = [[row]]
    with open(filename, 'a') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(rows)

r = 10

for i in range (0, 90):
    i = np.radians(i)
    x1 = (r + a)*np.cos(i)
    y1 = (r + a)*np.sin(i)
    # x.append(x1)
    # y.append(y1)
    x1 = str(x1)
    y1 = str(y1)
    row = s + " " + x1 + " " +  y1 + " " + c + " " + s1
    rows = [[row]]
    with open(filename, 'a') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(rows)

r = 5

for i in range (90, 180):
    i = np.radians(i)
    x1 = (r + a)*np.cos(i)
    y1 = 5 + (r + a)*np.sin(i)
    # x.append(x1)
    # y.append(y1)
    x1 = str(x1)
    y1 = str(y1)
    row = s + " " + x1 + " " +  y1 + " " + c + " " + s1
    rows = [[row]]
    with open(filename, 'a') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(rows)
