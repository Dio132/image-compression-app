
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import os
from tkinter import Tk, filedialog


def init_centroids(num_clusters, image):
    H, W, C = image.shape
    idx = np.random.randint(H * W, size=num_clusters)
    centroids_init = image[(idx // W).astype(int), idx % W].astype('float64')
    return centroids_init


def update_centroids(centroids, image, max_iter=30, print_every=10):
    H, W, C = image.shape
    idx = np.zeros((H, W))
    for it in range(max_iter):
        if (it + 1) % print_every == 0:
            print(f'Iteration {it + 1}/{max_iter}')
        for i in range(H):
            for j in range(W):
                idx[i, j] = np.argmin(np.linalg.norm(centroids - image[i, j], axis=1))
        for k in range(centroids.shape[0]):
            group = image[idx == k]
            if group.shape[0] > 0:
                centroids[k] = group.mean(axis=0)
    return centroids.astype(int)


def update_image(image, centroids):
    H, W, C = image.shape
    for i in range(H):
        for j in range(W):
            image[i, j] = centroids[np.argmin(np.linalg.norm(centroids - image[i, j], axis=1))]
    return image


def main():
    # Prompt user to choose an image file
    Tk().withdraw()  # Hide the root window
    filepath = filedialog.askopenfilename(title="Select an image file")

    if not filepath:
        print("No image selected. Exiting.")
        return

    image = mpimg.imread(filepath)
    image = np.copy(image)
    image.setflags(write=1)

    print('[INFO] Image loaded:', filepath)

    # Parameters
    num_clusters = 16
    max_iter = 100
    print_every = 10
    
    print("in proggress plz wait....")

    # K-means
    centroids = init_centroids(num_clusters, image)
    centroids = update_centroids(centroids, image, max_iter, print_every)
    compressed = update_image(image, centroids)

    # Save and show
    os.makedirs("outputs", exist_ok=True)
    plt.imshow(compressed)
    plt.axis("off")
    plt.title("Compressed Image")
    plt.savefig("outputs/compressed.png", bbox_inches='tight', transparent=True)
    plt.show()

    print("[INFO] Compression complete. Saved to outputs/compressed.png")


if __name__ == '__main__':
    main()
