from tkinter import X
import numpy as np

def load_data():
    data = np.loadtxt("data/houses.txt", delimiter=',')
    size = data[:,0]
    numRooms = data[:,1]
    numFloors = data[:,2]
    antiquity = data[:,3]
    price = data[:,4]
    return size, numRooms, numFloors, antiquity, price

def load_data_multi():
    data = np.loadtxt("data/houses.txt", delimiter=',')
    x = data[:,:-1]
    y = data[:,-1:]
    return x, y
