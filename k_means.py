import argparse
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.image import imread             #nepotrebno ovde
import os
import random           #nepotrebno


def init_centroids(num_clusters, image):
    """
    Initialize a `num_clusters` x image_shape[-1] nparray to RGB
    values of randomly chosen pixels of `image`

    Parameters
    ----------
    num_clusters : int
        Number of centroids/clusters
    image : nparray
        (H, W, C) image represented as an nparray

    Returns
    -------
    centroids_init : nparray
        Randomly initialized centroids
    """

    # *** START YOUR CODE ***
    H,W,C=image.shape                                                   #generise rand tacke po 'matrici' HxW, tj.
    randx = np.random.randint(H-1, size=num_clusters)                   #random inicijalizacija tacaka iz slike
    randy=np.random.randint(W-1,size=num_clusters)
    centroids_init = image[randx,randy].astype(float)    #castuvanje u float zbog daljih operacija
    # raise NotImplementedError('init_centroids function not implemented')
    # *** END YOUR CODE ***

    return centroids_init


def update_centroids(centroids, image, max_iter=30, print_every=10):
    """
    Carry out k-means centroid update step `max_iter` times

    Parameters
    ----------
    centroids : nparray
        The centroids stored as an nparray
    image : nparray
        (H, W, C) image represented as an nparray
    max_iter : int
        Number of iterations to run
    print_every : int
        Frequency of status update

    Returns
    -------
    new_centroids : nparray
        Updated centroids
    """
    # *** START YOUR CODE ***
    H, W, C = image.shape                                           #funkcija linalng norm se moze koristiti za distance
    new_centroids=np.zeros(centroids.shape)
    carr = np.zeros((H, W))        #za racunanje c^{(i)}
    iter = 0
    while iter < max_iter:
        iter+=1
        for i in range(H):
            for j in range(W):                                    #ord=2 zato sto je l2 norma 
                carr[i][j] = np.argmin(np.linalg.norm(image[i][j] - centroids, axis=1,ord=2)**2)     #obrazac iz materijala za c^{(i)} gde centroids niz predstavlja prvobitno  
                                                                                            #niz random tacaka      niz carr sadrzi informaciju o tome u kojem centroidu je i,j tacka
        for k in range(centroids.shape[0]):
            centroids[k] = image[carr==k].mean(axis=0)     #srednje vrednosti iz svake grupe cuvamo u nizu centroids i tako ga updateujemo->elementu iz image[] koji pripada k-tom centroidu se
        if iter%print_every==0:                              #racuna srednja vrednost->carr je matrica velicine iste kao image, carr[i][j] ukazuje kojem centroidu image[i][j] pripada
            print("Trenutna iteracija je:",iter)
    new_centroids=centroids                            
    # raise NotImplementedError('update_centroids function not implemented')
        # Usually expected to converge long before `max_iter` iterations
                # Initialize `dist` vector to keep track of distance to every centroid
                # Loop over all centroids and store distances in `dist`
                # Find closest centroid and update `new_centroids`
        # Update `new_centroids`
    # *** END YOUR CODE ***
    return new_centroids

def update_image(image, centroids):
    """
    Update RGB values of pixels in `image` by finding
    the closest among the `centroids`

    Parameters
    ----------
    image : nparray
        (H, W, C) image represented as an nparray
    centroids : int
        The centroids stored as an nparray

    Returns
    -------
    image : nparray
        Updated image
    """

    # *** START YOUR CODE ***
    H, W, C = image.shape
    for i in range(H):
        for j in range(W):
            image[i][j] = centroids[np.argmin(np.linalg.norm(image[i][j] - centroids, axis=1,ord=2)**2)]       #[i][j] piksel iz slike uzima svojstva boje od njemu dodeljenog centroida
    # raise NotImplementedError('update_image function not implemented')
            # Initialize `dist` vector to keep track of distance to every centroid
            # Loop over all centroids and store distances in `dist`
            # Find closest centroid and update pixel value in `image`
    # *** END YOUR CODE ***

    return image


def main(args):                             

    # Setup
    max_iter = args.max_iter
    print_every = args.print_every
    image_path_small = args.small_path
    image_path_large = args.large_path
    num_clusters = args.num_clusters
    figure_idx = 0

    # Load small image
    image = np.copy(mpimg.imread(image_path_small))
    print('[INFO] Loaded small image with shape: {}'.format(np.shape(image)))
    plt.figure(figure_idx)
    figure_idx += 1
    plt.imshow(image)
    plt.title('Original small image')
    plt.axis('off')
    savepath = os.path.join('.', 'orig_small.png')
    plt.savefig(savepath, transparent=True, format='png', bbox_inches='tight')

    # Initialize centroids
    print('[INFO] Centroids initialized')
    centroids_init = init_centroids(num_clusters, image)

    # Update centroids
    print(25 * '=')
    print('Updating centroids ...')
    print(25 * '=')
    centroids = update_centroids(centroids_init, image, max_iter, print_every)

    # Load large image
    image = np.copy(mpimg.imread(image_path_large))
    image.setflags(write=1)
    print('[INFO] Loaded large image with shape: {}'.format(np.shape(image)))
    plt.figure(figure_idx)
    figure_idx += 1
    plt.imshow(image)
    plt.title('Original large image')
    plt.axis('off')
    savepath = os.path.join('.', 'orig_large.png')
    plt.savefig(fname=savepath, transparent=True, format='png', bbox_inches='tight')

    # Update large image with centroids calculated on small image
    print(25 * '=')
    print('Updating large image ...')
    print(25 * '=')
    image_clustered = update_image(image, centroids)

    plt.figure(figure_idx)
    figure_idx += 1
    plt.imshow(image_clustered)
    plt.title('Updated large image')
    plt.axis('off')
    savepath = os.path.join('.', 'updated_large.png')
    plt.savefig(fname=savepath, transparent=True, format='png', bbox_inches='tight')

    print('\nCOMPLETE')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--small_path', default='./peppers-small.tiff',
                        help='Path to small image')
    parser.add_argument('--large_path', default='./peppers-large.tiff',
                        help='Path to large image')
    parser.add_argument('--max_iter', type=int, default=250,
                        help='Maximum number of iterations')
    parser.add_argument('--num_clusters', type=int, default=16,
                        help='Number of centroids/clusters')
    parser.add_argument('--print_every', type=int, default=10,
                        help='Iteration print frequency')
    args = parser.parse_args()
    main(args)
