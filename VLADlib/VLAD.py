import numpy as np
import itertools
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import BallTree
import pickle
import glob
import cv2
from sklearn.decomposition import PCA
from VLADlib.Descriptors import *
from tqdm import tqdm
import multiprocessing
import os



def getDescriptors(path):
	descriptors = []
	for imagePath in os.listdir(path):
		im = cv2.imread(path+'/'+imagePath)
		kp,des = describeORB(im)
		if des is not None:
			descriptors.append(des)
	descriptors = list(itertools.chain.from_iterable(descriptors))
	descriptors = np.asarray(descriptors)
	return descriptors

def kMeansDictionary(training, k):

    #K-means algorithm
    # est = KMeans(n_clusters=k, init='k-means++', tol=0.0001, verbose=1).fit(training)
    est = MiniBatchKMeans(n_clusters=k, init='k-means++', max_iter=100, batch_size=100, verbose=1,tol=0.0).fit(training)
    return est

def getVLADDescriptors(path, visualDictionary):
    descriptors = list()
    idImage = list()
    for imagePath in glob.glob(path + "/*.jpg"):

        print(imagePath)
        im = cv2.imread(imagePath)
        kp, des = describeORB(im)
        if des is not None:
            v = VLAD(des, visualDictionary)
            descriptors.append(v)
            idImage.append(imagePath)

    #list to array
    descriptors = np.asarray(descriptors)
    return descriptors, idImage

def VLAD(X, visualDictionary):
    predictedLabels = visualDictionary.predict(X)
    centers = visualDictionary.cluster_centers_
    labels = visualDictionary.labels_

    k = visualDictionary.n_clusters

    m, d = X.shape
    V = np.zeros([k,d])

    # for all the clusters (visual words)
    for i in range(k):
        # if there is at least one descriptor in that cluster
        if np.sum(predictedLabels==i) > 0:
            # add the diferences
            V[i] = np.sum(X[predictedLabels==i,:] - centers[i], axis=0)


    V = V.flatten()
    # power normalization, also called square-rooting normalization
    V = np.sign(V)*np.sqrt(np.abs(V))

    # L2 normalization
    V = V/np.sqrt(np.dot(V,V))
    print(len(V))
    return V



#Implementation of a improved version of VLAD
#reference: Revisiting the VLAD image representation
def improvedVLAD(X, visualDictionary):
    predictedLabels = visualDictionary.predict(X)
    centers = visualDictionary.cluster_centers_
    labels = visualDictionary.labels_
    k = visualDictionary.n_clusters

    m, d = X.shape
    V = np.zeros([k,d])
    #computing the differences

    # for all the clusters (visual words)
    for i in range(k):
        # if there is at least one descriptor in that cluster
        if np.sum(predictedLabels ==i ) > 0:
            # add the diferences
            V[i] = np.sum(X[predictedLabels==i,:]-centers[i],axis=0)


    V = V.flatten()
    # power normalization, also called square-rooting normalization
    V = np.sign(V)*np.sqrt(np.abs(V))

    # L2 normalization

    V = V/np.sqrt(np.dot(V,V))
    return V

def indexBallTree(X,leafSize):
    tree = BallTree(X, leaf_size=leafSize)
    return tree

def query(image, k, visualDictionary,tree):
    #read image
    im = cv2.imread(image)
    #compute descriptors
    kp, descriptor = describeORB(im)

    #compute VLAD
    
    v = VLAD(descriptor,visualDictionary)
    # v = improvedVLAD(descriptor,visualDictionary)

    #find the k most relevant images

    dist, ind = tree.query(v.reshape(1, -1), k)

    return dist, ind
