# Cluster feature to make visual words

from VLADlib.VLAD import *
from VLADlib.Descriptors import *
import argparse
import glob
import cv2

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
from sklearn.preprocessing import StandardScaler
#parser
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--descriptorsPath", required = True,
	help = "Path to the file that contains the descriptors")
ap.add_argument("-w", "--numberOfVisualWords", required = True,
	help = "number of visual words or clusters to be computed")
ap.add_argument("-o", "--output", required = True,
	help = "Path to where the computed visualDictionary will be stored")
args = vars(ap.parse_args())

#args
path = args["descriptorsPath"]
k = int(args["numberOfVisualWords"])
output = args["output"]

#computing the visual dictionary
print("estimating a visual dictionary of size: "+ str(k) + " for descriptors in path:" + path)


with open(path, 'rb') as f:
    descriptors = pickle.load(f)

visualDictionary = kMeansDictionary(descriptors, k)

#output
file = output + ".pickle"

with open(file, 'wb') as f:
	pickle.dump(visualDictionary, f)

print("The visual dictionary  is saved in " + file)

