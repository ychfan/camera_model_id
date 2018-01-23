import cv2
from tqdm import tqdm

filelist = '/ifp/users/haichao/projects/camid/data/dresden/filelist_train'

with open(filelist) as f:
    lines = f.read().splitlines()

for i in tqdm(range(len(lines))):
    line = lines[i]
    img = cv2.imread(line.split(' ')[0])
    if img == None:
        print(line)
