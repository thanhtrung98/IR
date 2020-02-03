# VLAD

This is an extended VLAD implementation based on the original implementation 
https://github.com/jorjasso/VLAD
# Test
download: https://drive.google.com/drive/folders/1553GwSToiMzMKPunY5-ZNsOPZ9N60yt4?usp=sharing
and add to folder tmp
run python query.py  -q querry/1.jpg -r 11 -dV tmp/visualDic.pickle -i tmp/balindex.pickle
## Computing VLAD features for a new dataset
Example VLAD with ORB descriptors with a visual dictionary with 2 visual words and an a ball tree as index. (Of course, 2 visual words is not useful, instead,  try 16, 32, 64, or 256 visual words)

dowload dataset: https://www.robots.ox.ac.uk/~vgg/data/oxbuildings/
and decompress and add to foler ./dataset

1. compute descriptors from a dataset. The supported descriptors are ORB, SIFT and SURF:
	```python
	python describe.py --dataset dataset  --output output
	```
	*Example
	```python
	python describe.py --dataset dataset --output tmp/descriptor_5K
	```

2.  Construct a visual dictionary from the descriptors in path -d, with -w visual words:
	```python
	python visualDictionary.py  -d descriptorPath -w numberOfVisualWords -o output
	```
	*Example :
	```python
	python visualDictionary.py -d tmp/descriptor_5K.pickle  -w 64 -o tmp/visualDic
	```

3. Compute VLAD descriptors from the visual dictionary:
	```python
	python vladDescriptors.py  -d dataset -dV visualDictionaryPath  -o output
	```
	*Example :
	```python
	python vladDescriptors.py  -d dataset -dV tmp/visualDic.pickle -o vlad_des
	```
	
4.  Make an index from VLAD descriptors using  a ball-tree DS:
	```python
	python indexBallTree.py  -d VLADdescriptorPath -l leafSize -o output
	```
	*Example :
	```python
	python indexBallTree.py  -d tmp/vlad_des.pickle -l 40 -o tmp/ballindex
	```

5. Query:
	```python
	python query.py  --query image  --index indexTree --retrieve retrieve
	```
        *Example
        ```python
        python query.py  -q querry/1.jpg -r 11 -dV tmp/visualDic.pickle -i tmp/balindex.pickle
        ```

Note: file rank.py : compute map to evaluation (error)
update come soon 
## Installation

First install conda , then:

```python
conda create --name vlad numpy scipy scikit-learn matplotlib python=3
source activate vlad
conda install -c menpo opencv3=3.1.0
```



