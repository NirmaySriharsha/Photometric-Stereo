# ##################################################################### #
# 16720: Computer Vision Homework 6
# Carnegie Mellon University
# April 12, 2022
# ##################################################################### #

import numpy as np
import matplotlib.pyplot as plt
from q1 import loadData, estimateAlbedosNormals, displayAlbedosNormals, estimateShape, plotSurface 
from q1 import estimateShape
from utils import enforceIntegrability, plotSurface
from matplotlib import cm
import cv2

def estimatePseudonormalsUncalibrated(I):

    """
    Question 2 (b)

    Estimate pseudonormals without the help of light source directions. 

    Parameters
    ----------
    I : numpy.ndarray
        The 7 x P matrix of loaded images

    Returns
    -------
    B : numpy.ndarray
        The 3 x P matrix of pseudonormals
    
    L : numpy.ndarray
        The 3 x 7 array of lighting directions

    """

    B = None
    L = None
    # Your code here
    u, sigma, v = np.linalg.svd(I, full_matrices = False)
    L = (u[:, 0:3].dot(np.sqrt(np.diag(sigma[0:3])))).T
    B = np.sqrt(np.diag(sigma[0:3])).dot(v[0:3, :])
    L = (u[:, 0:3].dot(np.diag(sigma[0:3]))).T
    B = v[0:3, :]
    #L = np.diag(sigma[0:3]).dot(u[0:3, :])
    #B = v[0:3, :]
    #sigma[3:] = 0
    #L = u.dot(np.diag(np.sqrt(sigma))).T[0:3]
    #B = np.sqrt(np.diag(sigma)).dot(v)[0:3]
    #print(L.shape)
    #print(B.shape)
    #L = u[:, 0:3].T
    #B = v[0:3, ]
    return B, L

def plotBasRelief(B, mu, nu, lam):

    """
    Question 2 (f)

    Make a 3D plot of of a bas-relief transformation with the given parameters.

    Parameters
    ----------
    B : numpy.ndarray
        The 3 x P matrix of pseudonormals

    mu : float
        bas-relief parameter

    nu : float
        bas-relief parameter
    
    lambda : float
        bas-relief parameter

    Returns
    -------
        None

    """

    # Your code here
    G = np.array([[1, 0, 0], [0, 1, 0], [mu, nu, lam]])
    B_new = np.linalg.inv(G.T).dot(B)
    albedos, normals = estimateAlbedosNormals(B_new)
    surface = estimateShape(normals, s)
    plotSurface(surface)

    albedos, normals = estimateAlbedosNormals(B_hat)
    normals = enforceIntegrability(normals, s)
    G = np.array([[1, 0, 0], [0, 1, 0], [mu, nu, lam]])
    normals = np.linalg.inv(G.T).dot(normals)
    surface = estimateShape(normals, s)
    plotSurface(surface)



if __name__ == "__main__":

    # Part 2 (b)
    # Your code here
    I, L, s = loadData()
    B_hat, L_hat = estimatePseudonormalsUncalibrated(I)
    #B_hat = enforceIntegrability(B_hat,s)
    #G = np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]])
    #B_hat = np.linalg.inv(G.T).dot(B_hat)
    albedos, normals = estimateAlbedosNormals(B_hat)
    albedoIm, normalIm = displayAlbedosNormals(albedos, normals, s)
    plt.imsave('2b-a.png', albedoIm, cmap = 'gray')
    plt.imsave('2b-b.png', normalIm, cmap = 'rainbow')
    #plt.imshow(albedoIm, cmap = 'gray')
    #plt.show()
    #plt.imshow(normalIm, cmap = 'rainbow')
    #plt.show()

    #print(L)
    #print(L_hat)
    # Part 2 (d)
    # Your code here
    #surface = estimateShape(normals, s)
    #plotSurface(surface)
    # Part 2 (e)
    # Your code here
    normals = enforceIntegrability(normals, s)
    #G = np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]])
    #normals = np.linalg.inv(G.T).dot(normals)
    #surface = estimateShape(normals, s)
    #plotSurface(surface)


    # Part 2 (f)
    # Your code here
    plotBasRelief(B_hat, -10, 100, 1)

