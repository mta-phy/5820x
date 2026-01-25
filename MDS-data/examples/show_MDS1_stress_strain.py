import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from mdsdata import MDS1



def main():
    strain, stress = MDS1.load_data(temperature=600, return_X_y=True)
    print(strain.shape)
    plt.scatter(strain, stress)
    plt.show()


    data = MDS1.load_data(temperature=600, as_frame=True)
    print(data.frame)
    

if __name__ == '__main__':
    main()