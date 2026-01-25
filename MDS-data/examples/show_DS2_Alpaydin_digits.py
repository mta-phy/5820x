import matplotlib.pyplot as plt
import numpy as np
from mdsdata import DS2_light, load_Alpaydin_digits



def main():
    images, labels = DS2_light().load_data(return_X_y=True)
    images, labels = load_Alpaydin_digits()
    
    print("number of images:", images.shape[0])
    print("number of images per digit: ", end='')
    print(np.histogram(labels, np.arange(-0.5, 10.5, 1))[0])


    fig, axes = plt.subplots(ncols=5, nrows=3, figsize=(7, 5))
    rng = np.random.default_rng()
    for idx, ax in enumerate(axes.ravel()):
        idx = rng.integers(low=0, high=images.shape[0])

        ax.imshow(images[idx], cmap='gray', vmin=0, vmax=255)
        ax.set(xticks=[], yticks=[], title=f"{labels[idx]}")
    plt.show()



if __name__ == '__main__':
    main()
