# import the necessary packages
import numpy as np
import argparse
import imutils
import time
import cv2
import os
import glob

files = glob.glob('output/*.png')
for f in files:
	os.remove(f)

from sort import *
tracker = Sort()
memory = {}
line = [(217, 444), (964, 409)]
counter = 0
fronthuman = 0
backhuman = 0
BACK = 0
FRONT = 0
k=False
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
	help="path to input video")
ap.add_argument("-o", "--output", required=True,
	help="path to output video")
ap.add_argument("-y", "--yolo", required=True,
	help="base path to YOLO directory")
ap.add_argument("-c", "--confidence", type=float, default=0.3,
	help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.5,
	help="threshold when applyong non-maxima suppression")
args = vars(ap.parse_args())
print(args['confidence'])
print(args['threshold'])
# Return true if line segments AB and CD intersect
def intersect(A,B,C,D):
	return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

def ccw(A,B,C):
	return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

# load the COCO class labels our YOLO model was trained on
labelsPath = os.path.sep.join([args["yolo"], "classes.names"])
LABELS = open(labelsPath).read().strip().split("\n")
print(LABELS)
# initialize a list of colors to represent each possible class label
np.random.seed(42)
font = cv2.QT_FONT_NORMAL
COLORS = np.random.randint(0, 255, size=(200, 3),
	dtype="uint8")

# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join([args["yolo"], "yolov3_custom_10000.weights"])
configPath = os.path.sep.join([args["yolo"], "yolov3_custom.cfg"])

# load our YOLO object detector trained on COCO dataset (80 classes)
# and determine only the *output* layer names that we need from YOLO
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNet(weightsPath,configPath)
ln = net.getUnconnectedOutLayersNames()
# ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# initialize the video stream, pointer to output video file, and
# frame dimensions
vs = cv2.VideoCapture(args["input"])
writer = None
# (W, H) = (None, None)

frameIndex = 0

# try to determine the total number of frames in the video file
try:
	prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
		else cv2.CAP_PROP_FRAME_COUNT
	total = int(vs.get(prop))
	print("[INFO] {} total frames in video".format(total))

# an error occurred while trying to determine the total
# number of frames in the video file
except:
	print("[INFO] could not determine # of frames in video")
	print("[INFO] no approx. completion time can be provided")
	total = -1

# loop over frames from the video file stream
while True:
	# read the next frame from the file
	grabbed, frame = vs.read()
	H, W ,_= frame.shape
	print("start")
	# if the frame was not grabbed, then we have reached the end
	# of the stream
	if not grabbed:
		break

	# if the frame dimensions are empty, grab them
	# if W is None or H is None:
	# 	(H, W) = frame.shape

	# construct a blob from the input frame and then perform a forward
	# pass of the YOLO object detector, giving us our bounding boxes
	# and associated probabilities
	blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), (0,0,0) ,swapRB=True, crop=False)
	net.setInput(blob)
	start = time.time()
	layerOutputs = net.forward(ln)
	end = time.time()

	# initialize our lists of detected bounding boxes, confidences,
	# and class IDs, respectively
	boxes = []
	confidences = []
	classIDs = []
	labelr = []
	# loop over each of the layer outputs
	for output in layerOutputs:
		# loop over each of the detections
		for detection in output:
			# extract the class ID and confidence (i.e., probability)
			# of the current object detection
			scores = detection[5:]
			classID = np.argmax(scores)
			confidence = scores[classID]

			# filter out weak predictions by ensuring the detected
			# probability is greater than the minimum probability
			if confidence > args["confidence"]:
				print("5555555555")
				centerX = int(detection[0]*W)
				centerY = int(detection[1]*H)
				width = int(detection[2]*W)
				height = int(detection[3]*H)
				# scale the bounding box coordinates back relative to
				# the size of the image, keeping in mind that YOLO
				# actually returns the center (x, y)-coordinates of
				# the bounding box followed by the boxes' width and
				# height
				
				# box = detection[0:4] * np.array([W, H, W, H])
				# (centerX, centerY, width, height) = box.astype("int")

				# use the center (x, y)-coordinates to derive the top
				# and and left corner of the bounding box
				x = int(centerX - (width / 2))
				y = int(centerY - (height / 2))

				# update our list of bounding box coordinates,
				# confidences, and class IDs
				boxes.append([x, y, int(width), int(height)])
				confidences.append(float(confidence))
				classIDs.append(classID)
	colors = np.random.uniform(0,255,size=(len(boxes),3))
	
	# apply non-maxima suppression to suppress weak, overlapping
	# bounding boxes
	idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"], args["threshold"])
	# for i in idxs.flatten():
	# 	x5,y5,w5,h5 = boxes[i]
	# 	label =str(LABELS[classIDs[i]])
	# 	# print(label)
	# 	labelr.append(label)
	# 	confidence = str(round(confidences[i],1))
	# 	color = colors[i]
	# 	cv2.rectangle(frame,(x5,y5),(x5+w5 , y5+h5), color, 2)
	# 	cv2.putText(frame, label + " ", (x5, y5+20), font, 0.5, (255,255,255),2)
	dets = []
	dets2 = []
	if len(idxs) > 0:
		# loop over the indexes we are keeping
		for i in idxs.flatten():
			x5,y5,w5,h5 = boxes[i]
			label =str(LABELS[classIDs[i]])
			labelr.append(label)
			confidence = str(round(confidences[i],1))
			color = colors[i]
			cv2.rectangle(frame,(x5,y5),(x5+w5 , y5+h5), color, 2)
			cv2.putText(frame, label + " ", (x5, y5+20), font, 0.5, (255,255,255),2)
			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])
			dets.append([x, y, x+w, y+h])
			p5 = (int(x + (w/2)), int(y + (h/2)))
			dets2.append([x, y, w, h])
			# cv2.line(frame, (x,y), (w,h), color, 3)
			# print(labelr)
			# print(dets)
	np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
	dets = np.asarray(dets)
	tracks = tracker.update(dets)
	dets1 = dets[::-1]
	print(dets)
	print(dets2)
	print(tracks)
	boxes = []
	boxes1 = []
	indexIDs = []
	c = []
	previous = memory.copy()
	memory = {}
	u= len(labelr)
	# u=0
	u=len(dets1)
	for track,de in zip(dets1,dets1):
		boxes.append([track[0], track[1], track[2], track[3]])
		boxes1.append([de[0], de[1], de[2], de[3]])
		if u>0:
			indexIDs.append(int(u))
		memory[indexIDs[-1]] = boxes1[-1]
		u=u-1
		print("index",memory)
	if len(boxes) > 0:
		i = int(0)
		
		for box in boxes:

			# extract the bounding box coordinates
			(x, y) = (int(box[0]), int(box[1]))
			(w, h) = (int(box[2]), int(box[3]))
			print("sssss",x,y,w,h)
			# draw a bounding box rectangle and label on the image
			# color = [int(c) for c in COLORS[classIDs[i]]]
			# cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
			
			color = [int(c) for c in COLORS[indexIDs[i] % len(COLORS)]]
			# cv2.rectangle(frame, (x, y), (w, h), color, 2)
			if indexIDs[i] in previous:
				previous_box = memory[indexIDs[i]]
				(x2, y2) = (int(previous_box[0]), int(previous_box[1]))
				(w2, h2) = (int(previous_box[2]), int(previous_box[3]))
				# p0 = (int(x + (w-x)/2), int(y + (h-y)/2))
				# p1 = (int(x2 + (w2-x2)/2), int(y2 + (h2-y2)/2))
				p0 = (int((x + (w-x)/2)-2), int((y + (h-y)/2)+2))
				p1 = (int((x2 + (w2-x2)/2)+2), int((y2 + (h2-y2)/2)-2))
				cv2.line(frame, p0, p1, color, 10)
				
				print(indexIDs[i])
				print(previous)
				print(labelr[u-1])
				print((x,y),(x2,y2))
				print((w,h),(w2,h2))
				if intersect(p0, p1, line[0], line[1]):
					labelsss = labelr[u-1]
					if(labelsss == 'FRONT HUMAN'):
						fronthuman+=1
					elif(labelsss == 'FRONT'):
						FRONT+=1
					elif(labelsss == 'BACK'):
						BACK+=1
					elif(labelsss == 'BACK HUMAN'):
						backhuman+=1	
					counter += 1
				u=u-1
			text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
			# text = "{}".format(indexIDs[i])
			# cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
			i += 1

	# draw line
	cv2.line(frame, line[0], line[1], (0, 255, 255), 3)
	

	# draw counter
	# cv2.putText(frame, str(counter), (100,200), cv2.FONT_HERSHEY_DUPLEX, 5.0, (0, 255, 255), 10)
	cv2.putText(frame, 'FRONT'+ ":" + str(FRONT), (20, 20), font, 0.8, (0,0,0),2)
	cv2.putText(frame, 'FRONT HUMAN'+ ":" + str(fronthuman), (20, 50), font, 0.8, (0,0,0),2)
	cv2.putText(frame, 'BACK'+ ":" + str(BACK), (20, 80), font, 0.8, (0,0,0),2)
	cv2.putText(frame, 'BACK HUMAN'+ ":" + str(backhuman), (20, 110), font, 0.8, (0,0,0),2)
	# counter += 1

	# saves image file
	cv2.imwrite("output/frame-{}.png".format(frameIndex), frame)

	# check if the video writer is None
	if writer is None:
		# initialize our video writer
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(args["output"], fourcc, 30,
			(frame.shape[1], frame.shape[0]), True)

		# some information on processing single frame
		if total > 0:
			elap = (end - start)
			print("[INFO] single frame took {:.4f} seconds".format(elap))
			print("[INFO] estimated total time to finish: {:.4f}".format(
				elap * total))

	# write the output frame to disk
	writer.write(frame)

	# increase frame index
	frameIndex += 1

	#if frameIndex >= 4000: # limits the execution to the first 4000 frames
	#	print("[INFO] cleaning up...")
	#	writer.release()
	#	vs.release()
	#	exit()

# release the file pointers
print("[INFO] cleaning up...")
writer.release()
vs.release()