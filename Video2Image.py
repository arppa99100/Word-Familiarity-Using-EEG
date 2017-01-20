import cv2
import os
import math
source = r"/media/tauqs/0D0A0EE30D0A0EE3/Tauqueer/Academics/7th Sem/Project/Ver-2/theta"


for video in os.listdir(source):
	cap = cv2.VideoCapture(source+"/"+video)
	
	number_frames = cap.get(7)
	total_frames = math.floor(number_frames/25)
	frames_remove = number_frames-total_frames*25

	print number_frames
	print cap.get(5)
	print video+" :",12.0/cap.get(5)
	print number_frames,cap.get(5),(number_frames*1.0/cap.get(5))
	print "----"
