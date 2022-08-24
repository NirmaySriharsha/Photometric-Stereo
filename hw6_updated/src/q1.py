# ##################################################################### #
# 16720: Computer Vision Homework 6
# Carnegie Mellon University
# April 12, 2022
###################################################################### #

import numpy as np
from matplotlib import pyplot as plt
from skimage.color import rgb2xyz
from utils import integrateFrankot, plotSurface
import cv2

def renderNDotLSphere(center, rad, light, pxSize, res):

    """
    Question 1 (b)

    Render a hemispherical bowl with a given center and radius. Assume that
    the hollow end of the bowl faces in the positive z direction, and the
    camera looks towards the hollow end in the negative z direction. The
    camera's sensor axes are aligned with the x- and y-axes.

    Parameters
    ----------
    center : numpy.ndarray
        The center of the hemispherical bowl in an array of size (3,)

    rad : float
        The radius of the bowl

    light : numpy.ndarray
        The direction of incoming light

    pxSize : float
        Pixel size

    res : numpy.ndarray
        The resolution of the camera frame

    Returns
    -------
    image : numpy.ndarray
        The rendered image of the hemispherical bowl
    """

    [X, Y] = np.meshgrid(np.arange(res[0]), np.arange(res[1]))
    X = ((X - res[0]/2) * pxSize + center[0])*1.e-4
    Y = ((Y - res[1]/2) * pxSize + center[1])*1.e-4
    Z = np.sqrt(rad**2+0j-X**2-Y**2)
    X[np.real(Z) == 0] = 0
    Y[np.real(Z) == 0] = 0
    Z = np.real(Z)

    image = None
    # Your code here
    dots = np.stack((X, Y, Z), axis = 2).reshape((res[0]*res[1], -1))
    dots = (dots.T/ np.linalg.norm(dots, axis = 1).T).T
    image = np.dot(dots, light).reshape((res[1], res[0]))
    image[Z==0] = 0
    return image


def loadData(path = "./data/"):

    """
    Question 1 (c)

    Load data from the path given. The images are stored as input_n.tif
    for n = {1...7}. The source lighting directions are stored in
    sources.mat.

    Parameters
    ---------
    path: str
        Path of the data directory

    Returns
    -------
    I : numpy.ndarray
        The 7 x P matrix of vectorized images

    L : numpy.ndarray
        The 3 x 7 matrix of lighting directions

    s: tuple
        Image shape

    """

    I = None
    L = None
    s = None
    # Your code here
    for i in range(7):
        imagepath = path + "input_{}.tif".format(i+1)
        image = cv2.imread(imagepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if I is None: 
            height, width = image.shape
            P = height*width
            I = np.zeros((7, P))
        I[i, :] = np.reshape(image, (1, P))
    s = (height, width)
    L = np.load(path + "sources.npy").T

    return I, L, s


def estimatePseudonormalsCalibrated(I, L):

    """
    Question 1 (e)

    In calibrated photometric stereo, estimate pseudonormals from the
    light direction and image matrices

    Parameters
    ----------
    I : numpy.ndarray
        The 7 x P array of vectorized images

    L : numpy.ndarray
        The 3 x 7 array of lighting directions

    Returns
    -------
    B : numpy.ndarray
        The 3 x P matrix of pesudonormals
    """

    B = None
    # Your code here
    B = np.linalg.inv(np.dot(L, L.T)).dot(L).dot(I)
    return B


def estimateAlbedosNormals(B):

    '''
    Question 1 (e)

    From the estimated pseudonormals, estimate the albedos and normals

    Parameters
    ----------
    B : numpy.ndarray
        The 3 x P matrix of estimated pseudonormals

    Returns
    -------
    albedos : numpy.ndarray
        The vector of albedos

    normals : numpy.ndarray
        The 3 x P matrix of normals
    '''

    albedos = None
    normals = None
    # Your code here
    albedos = np.linalg.norm(B, axis = 0)
    normals = B/(albedos + 1e-6)
    return albedos, normals


def displayAlbedosNormals(albedos, normals, s):

    """
    Question 1 (f, g)

    From the estimated pseudonormals, display the albedo and normal maps

    Please make sure to use the `coolwarm` colormap for the albedo image
    and the `rainbow` colormap for the normals.

    Parameters
    ----------
    albedos : numpy.ndarray
        The vector of albedos

    normals : numpy.ndarray
        The 3 x P matrix of normals

    s : tuple
        Image shape

    Returns
    -------
    albedoIm : numpy.ndarray
        Albedo image of shape s

    normalIm : numpy.ndarray
        Normals reshaped as an s x 3 image

    """

    albedoIm = None
    normalIm = None
    # Your code here
    albedoIm = np.reshape((albedos/np.max(albedos)), s)
    #normalIm = np.reshape(((normals+1.)/2.).T, (s[0], s[1], 3))
    normalIm = np.reshape(((normals+1)/2).T, (s[0], s[1], 3))
    return albedoIm, normalIm


def estimateShape(normals, s):

    """
    Question 1 (j)

    Integrate the estimated normals to get an estimate of the depth map
    of the surface.

    Parameters
    ----------
    normals : numpy.ndarray
        The 3 x P matrix of normals

    s : tuple
        Image shape

    Returns
    ----------
    surface: numpy.ndarray
        The image, of size s, of estimated depths at each point

    """

    surface = None
    # Your code here
    f_x = np.reshape(normals[0, :]/(-normals[2, :] + 1e-5), s)
    f_y = np.reshape(normals[1, :]/(-normals[2, :]) + 1e-5, s)
    surface = integrateFrankot(f_x, f_y)
    return surface

if __name__ == '__main__':
    # Part 1(b)
    radius = 0.75 # cm
    center = np.asarray([0, 0, 0]) # cm
    pxSize = 7 # um
    res = (3840, 2160)

    light = np.asarray([1, 1, 1])/np.sqrt(3)
    image = renderNDotLSphere(center, radius, light, pxSize, res)
    plt.figure()
    plt.imshow(image, cmap = 'gray')
    plt.imsave('1b-a.png', image, cmap = 'gray')

    light = np.asarray([1, -1, 1])/np.sqrt(3)
    image = renderNDotLSphere(center, radius, light, pxSize, res)
    plt.figure()
    plt.imshow(image, cmap = 'gray')
    plt.imsave('1b-b.png', image, cmap = 'gray')

    light = np.asarray([-1, -1, 1])/np.sqrt(3)
    image = renderNDotLSphere(center, radius, light, pxSize, res)
    plt.figure()
    plt.imshow(image, cmap = 'gray')
    plt.imsave('1b-c.png', image, cmap = 'gray')

    # Part 1(c)
    I, L, s = loadData('./data/')

    # Part 1(d)
    # Your code here
    _, values, _ = np.linalg.svd(I, full_matrices = False)
    print(values)

    # Part 1(e)
    B = estimatePseudonormalsCalibrated(I, L)

    # Part 1(f)
    albedos, normals = estimateAlbedosNormals(B)
    albedoIm, normalIm = displayAlbedosNormals(albedos, normals, s)
    plt.imsave('1f-a.png', albedoIm, cmap = 'gray')
    plt.imsave('1f-b.png', normalIm, cmap = 'rainbow')

    # Part 1(i)
    surface = estimateShape(normals, s)
    plotSurface(surface)
