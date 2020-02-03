from VLADlib.VLAD import *
from VLADlib.Descriptors import *
import itertools
import argparse
import glob
import cv2
import time
#parser
ap = argparse.ArgumentParser()
ap.add_argument("-q", "--query", required = True,
	help = "Path of a query image")
ap.add_argument("-r", "--retrieve", required = True,
	help = "number of images to retrieve")
ap.add_argument("-dV", "--visualDictionary", required = True,
	help = "Path to the visual dictionary")
ap.add_argument("-i", "--index", required = True,
	help = "Path of the Ball tree")

args = vars(ap.parse_args())

#args
path = args["query"]
k = int(args["retrieve"])

pathVD = args["visualDictionary"]
treeIndex = args["index"]
start = time.time()
#load the index
with open(treeIndex, 'rb') as f:
    indexStructure = pickle.load(f)

#load the visual dictionary
with open(pathVD, 'rb') as f:
    visualDictionary = pickle.load(f)     

imageID = indexStructure[0]
tree = indexStructure[1]
pathImageData = indexStructure[2]
rank=[]
#computing descriptors
dist, ind = query(path, k, visualDictionary,tree)

print("thoi gian",time.time()-start)
print(dist)
print(ind)
ind = list(itertools.chain.from_iterable(ind))
rank.append(ind)
print(rank)
# display the query
imageQuery = cv2.imread(path)

# loop over the results
for i in ind:
	# load the result image and display it
	result = cv2.imread(imageID[i])
	cv2.imshow("Result", result)
	cv2.waitKey(0)

