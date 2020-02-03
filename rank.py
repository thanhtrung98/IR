from VLADlib.VLAD import *
from VLADlib.Descriptors import *
import itertools
import argparse
import glob
import cv2
from evaluate import compute_map_and_print
import time
#parser
ap = argparse.ArgumentParser()
ap.add_argument("-dV", "--visualDictionary", required = True,
	help = "Path to the visual dictionary")
ap.add_argument("-i", "--index", required = True,
	help = "Path of the Ball tree")
ap.add_argument('--gnd_path',
                        type=str,
                        default="./gnd_oxford5k.pkl",
                        help="""
                        Path to ground-truth
                        """)
ap.add_argument('-n', '--truncation_size',
                        type=int,
                        default=1000,
                        help="""
                        Number of images in the truncated gallery
                        """)

args = vars(ap.parse_args())
def evaluate(ranks):
    gnd_name = os.path.splitext(os.path.basename('./gnd_oxford5k.pkl'))[0]
    with open('./gnd_oxford5k.pkl', 'rb') as f:
        gnd = pickle.load(f)['gnd']
    compute_map_and_print(gnd_name.split("_")[-1], ranks.T, gnd)
#args
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
rank = []
folder = './querry'
#computing descriptors
for path in sorted(os.listdir(folder)):
    print(folder+'/'+path)
    dist, ind = query(folder+'/'+path, 100, visualDictionary,tree)
    ind = list(itertools.chain.from_iterable(ind))
    rank.append(ind)
rank = np.asmatrix(rank)
print(rank)
evaluate(rank)
