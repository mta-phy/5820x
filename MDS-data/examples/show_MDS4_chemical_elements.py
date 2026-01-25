import matplotlib.pyplot as plt
from mdsdata import MDS4

""" Usage of the MDS-dataset 'MDS-4: Chemical Elements'

This script contains some examples for how to
use the MDS-dataset 'MDS-4: Chemical Elements'. For further
information and reference to the source of the data please
refer to the MDS-book.
"""

def main():
    # How to use the numpy version of the dataset
    databunch = MDS4.load_data(as_frame=True)
    df = databunch.frame

    print(df)
    print("the two classes are: 0:", databunch.target_names[0],  ", 1:", databunch.target_names[1])

    for label in [0, 1]:
        mask = df['target'] == label
        plt.scatter(x=df['atomic_radius'][mask], 
                    y=df['electronegativity'][mask],
                    c=f'C{label}', label=databunch.target_names[label])
    plt.legend()
    plt.xlabel('atomic_radius')
    plt.ylabel('electronegativity')
    plt.show()


if __name__ == '__main__':
    main()