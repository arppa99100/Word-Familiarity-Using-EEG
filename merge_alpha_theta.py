import os
import cv2
import numpy as np
source1 = r"/media/tauqs/0D0A0EE30D0A0EE3/Tauqueer/Academics/7th Sem/Project/Ver-2/Data1"
source2 = r"/media/tauqs/0D0A0EE30D0A0EE3/Tauqueer/Academics/7th Sem/Project/Ver-2/Data2"

os.chdir("./Data")
for folder1,folder2 in zip(os.listdir(source1),os.listdir(source2)):
	img_src1 = source1+'/'+folder1+'/TimedVersion'
	img_src2 = source2+'/'+folder2+'/TimedVersion'

	print 'folder1: '+folder1, 'folder2: '+folder2

	os.makedirs(folder1)
	os.chdir("./"+folder1)
	os.makedirs("TimedVersion")
	os.chdir("./TimedVersion")

	for image1,image2 in zip(os.listdir(img_src1),os.listdir(img_src2)):
		img1 = cv2.imread(img_src1+'/'+image1)
		img2 = cv2.imread(img_src2+'/'+image2)
		vis = np.concatenate((img1, img2), axis=1)
		small = cv2.resize(vis, (0,0), fx=0.5, fy=1) 
		cv2.imwrite(image1, small)

	os.chdir("..")
	os.chdir("..")