from VLADlib.VLAD import *
from VLADlib.Descriptors import *
import argparse
import glob
import cv2

#parser
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required = True,
	help = "Path to image dataset")
ap.add_argument("-dV", "--visualDictionaryPath", required = True,
	help = "Path to the visual dictionary")
ap.add_argument("-o", "--output", required = True,
	help = "Path to where VLAD descriptors will be stored")
args = vars(ap.parse_args())


#args
path = args["dataset"]
pathVD = args["visualDictionaryPath"]
output = args["output"]


#estimating VLAD descriptors for the whole dataset

with open(pathVD, 'rb') as f:
    visualDictionary = pickle.load(f) 


#computing the VLAD descriptors
V, idImages = getVLADDescriptors(path, visualDictionary)


#output
file = output + ".pickle"

with open(file, 'wb') as f:
	pickle.dump([idImages, V, path], f)

print("The VLAD descriptors are  saved in "+file)

