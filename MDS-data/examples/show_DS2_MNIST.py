import matplotlib.pyplot as plt
import numpy as np
from mdsdata import DS2



def main():
    images, labels = DS2().load_data(return_X_y=True, train=True)
    
    print("number of images:", images.shape[0])
    print("number of images per digit: ", end='')
    print(np.histogram(labels, np.arange(-0.5, 10.5, 1))[0])


    fig, axes = plt.subplots(ncols=5, nrows=3, figsize=(7, 5))
    rng = np.random.default_rng()
    for ax in axes.ravel():
        idx = rng.integers(low=0, high=images.shape[0])
        ax.imshow(images[idx], cmap='gray', vmin=0, vmax=255)
        ax.set(xticks=[], yticks=[], title=f"{labels[idx]}")
    plt.show()



if __name__ == '__main__':
    main()
