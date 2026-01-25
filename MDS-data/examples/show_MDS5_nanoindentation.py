import matplotlib.pyplot as plt
from mdsdata import MDS5

""" Usage of the MDS-dataset 'MDS-5: Nanoindentation'

This script contains some examples for how to use the MDS-dataset. For 
further information and reference to the source of the data please
refer to the MDS-book.
"""

def main():
    CuCr = MDS5.load_data()
    X = CuCr.feature_matrix
    y = CuCr.target 
    print("The feature matrix has", X.shape[1], "features in columns:", CuCr.feature_names)
    print(" ... and", X.shape[0], "data records as rows of X.")
    print("The class labels 0...3 of Y correspond to:", CuCr.target_names)

    modulus = X[:, 0]
    hardness = X[:, 1]
    material = y

    fig, ax = plt.subplots()
    ax.scatter(modulus, hardness, c=material)
    ax.set(xlabel="Young's modulus [GPa]", ylabel="hardness [GPa]")
    plt.show()


if __name__ == '__main__':
    main()