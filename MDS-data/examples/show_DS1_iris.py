import matplotlib.pyplot as plt
import numpy as np
from mdsdata import DS1

""" Usage of the MDS-dataset 'MDS-1: Iris Flower dataset'

This script contains some examples for how to use the MDS-dataset. For 
further information and reference to the source of the data please
refer to the MDS-book.
"""



def main():
    iris = DS1.load_data()
    X = iris.data 
    y = iris.target
    class_names = iris.target_names



    fig, ax = plt.subplots(dpi=100)

    for i in [0, 1, 2]:
        mask = (y == i)
        ax.scatter(X[mask,0], X[mask,1], c=y[mask], label=class_names[i], vmin=0, vmax=2)

    ax.set(xlabel='sepal length [cm]', ylabel='sepal width [cm]')
    ax.legend()
    plt.show()





if __name__ == '__main__':
    main()
