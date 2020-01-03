# Created on Jan 2nd 2020
# Author: Changzhi Cai
# Contact me: caichangzhi97@gmail.com

# import package
import argparse
import cv2

# OpenCV's tracking algorithm
OPENCV_OBJECT_TRACKERS = {
	"csrt":cv2.TrackerCSRT_create,
	"kcf":cv2.TrackerKCF_create,
	"boosting":cv2.TrackerBoosting_create,
	"mil":cv2.TrackerMIL_create,
	"tld":cv2.TrackerTLD_create,
	"medianflow":cv2.TrackerMedianFlow_create,
	"mosse":cv2.TrackerMOSSE_create
}

# set parameters
ap = argparse.ArgumentParser()
ap.add_argument("-t","--tracker",type = str,default = "kcf",help = "OpenCV object tracker type")
args = vars(ap.parse_args())

# instantiate OpenCV's multi-object tracker
trackers = cv2.MultiTracker_create()
vs = cv2.VideoCapture("soccer_01.mp4")

# video stream
while True:
    
    # take the current frame
    frame = vs.read()
    frame = frame[1]
    
    # end with video end
    if frame is None:
        break
    
    # resize each frame
    (h,w) = frame.shape[:2]
    width = 600
    r = width/float(w)
    dim = (width,int(h*r))
    frame = cv2.resize(frame,dim,interpolation = cv2.INTER_AREA)
    
    # tracking results
    (success,boxes) = trackers.update(frame)
    
    # plotting area
    for box in boxes:
        (x,y,w,h) = [int(v) for v in box]
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        
    # display
    cv2.imshow("Frame",frame)
    key = cv2.waitKey(100) & 0xFF
    
    if key == ord("s"):
        
        # choose an area and press 's'
        box = cv2.selectROI("Frame",frame,fromCenter = False,showCrosshair = True)
        
        # create a new tracker
        tracker = OPENCV_OBJECT_TRACKERS[args["tracker"]]()
        trackers.add(tracker,frame,box)
        
    # exit
    elif key == 27:
        break
    
vs.release()
cv2.destroyAllWindows()