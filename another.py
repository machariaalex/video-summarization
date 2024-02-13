import cv2
import numpy as np

video = cv2.VideoCapture(0)

# Get video properties
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create VideoWriters for both original and summarized videos
original_writer = cv2.VideoWriter('original_video.mp4', cv2.VideoWriter_fourcc(*'DIVX'), 25, (width, height))
summarized_writer = cv2.VideoWriter('summarized_video.mp4', cv2.VideoWriter_fourcc(*'DIVX'), 25, (width, height))

# Rest of the code remains unchanged
ret, frame1 = video.read()
prev_frame = frame1

a = 0
b = 0
c = 0
object_count = 0

protopath = 'MobileNetSSD_deploy.prototxt'
modelpath = 'MobileNetSSD_deploy.caffemodel'

detector = cv2.dnn.readNetFromCaffe(prototxt=protopath, caffeModel=modelpath)

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

while True:
    ret, frame = video.read()
    
    # Rest of the processing code...

    # Save original frame to original video
    original_writer.write(frame)

    cv2.imshow("Application", frame)
    c += 1

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

# Release resources after the loop ends
video.release()
original_writer.release()
summarized_writer.release()
cv2.destroyAllWindows()

print("Total frames: ", c)
print("Unique frames: ", a)
print("Common frames: ", b)
print("Object count: ", object_count)
