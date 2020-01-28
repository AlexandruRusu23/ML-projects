import matplotlib.pyplot as plt
import numpy as np
import os
from imutils import paths

def generate_figures(folder_name):
    fig = plt.figure()
    fig.canvas.set_window_title(folder_name)

    for index in range(0, 50):
        ax = fig.add_subplot(10, 5, index+1)
        image_paths = list(paths.list_images(folder_name+str(index)+"/"))
        hist_list = []
        for image_path in image_paths:
            label = image_path.split(os.path.sep)[-1].split(" ")[0]
            hist_list.append(int(label))

        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.hist(hist_list, bins=50)

def main():
    generate_figures("output_ward/")
    generate_figures("output_kmeans/")
    plt.show()

if __name__ == "__main__":
    main()
