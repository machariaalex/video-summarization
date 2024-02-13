import cv2
import numpy as np
import time

# Open a connection to the default camera (index 0)
video = cv2.VideoCapture(0)

# Get video properties
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create VideoWriters for both original and summarized videos
original_writer = cv2.VideoWriter('ori_video.mp4', cv2.VideoWriter_fourcc(*'DIVX'), 25, (width, height))
summarized_writer = cv2.VideoWriter('summaiz_video.mp4', cv2.VideoWriter_fourcc(*'DIVX'), 25, (width, height))

# Set the duration of webcam activation (90 seconds)
duration = 90  # seconds
end_time = time.time() + duration

# Create VideoWriter for recording events
event_writer = None

# Initialize variables
ret, frame1 = video.read()
prev_frame = frame1
a = 0
b = 0
c = 0
object_count = 0
people_count = 0
objects_in_motion = set()
fence_line_x = int(width * 0.2)  # Adjust the position of the fence line

# Load MobileNet SSD model
protopath = 'MobileNetSSD_deploy.prototxt'
modelpath = 'MobileNetSSD_deploy.caffemodel'
detector = cv2.dnn.readNetFromCaffe(prototxt=protopath, caffeModel=modelpath)

# Define object classes
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

while time.time() < end_time:
    # Read a frame from the camera
    ret, frame = video.read()

    # Perform object detection
    (H, W) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
    detector.setInput(blob)
    person_detections = detector.forward()

    # Process detected objects
    for i in np.arange(0, person_detections.shape[2]):
        confidence = person_detections[0, 0, i, 2]
        if confidence > 0.5:
            idx = int(person_detections[0, 0, i, 1])

            # Increment object_count when a person is detected
            if CLASSES[idx] == "person":
                people_count += 1
            else:
                objects_in_motion.add(CLASSES[idx])

            person_box = person_detections[0, 0, i, 3:7] * np.array([W, H, W, H])
            (startX, startY, endX, endY) = person_box.astype("int")

            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)

            # Save original frame to original video
            original_writer.write(frame)

            # Save summarized frame to summarized video
            if np.sum(np.absolute(frame - prev_frame)) / np.size(frame) > 20.:
                summarized_writer.write(frame)
                prev_frame = frame
                a += 1
            else:
                b += 1

            # Save original frame to event video
            if event_writer is None:
                event_writer = cv2.VideoWriter(f'event_{time.time()}.mp4', cv2.VideoWriter_fourcc(*'DIVX'), 25, (width, height))
            
            event_writer.write(frame)

    # Draw the virtual fence line
    cv2.line(frame, (fence_line_x, 0), (fence_line_x, height), (0, 255, 0), 2)

    # Check if an object crosses the virtual fence
    if startX < fence_line_x and endX > fence_line_x:
        # Capture an image and save as a .png
        cv2.imwrite(f'crossed_fence_{time.time()}.png', frame)

    # If there was no person detected, close the event_writer
    if event_writer is not None and people_count == 0:
        event_writer.release()
        event_writer = None

    # Display the frame
    cv2.imshow("Application", frame)
    c += 1

    # Check for the 'q' key to exit the loop
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

# Release resources after the loop ends
video.release()
original_writer.release()
summarized_writer.release()
cv2.destroyAllWindows()

# Save people count and objects in motion to a text file
with open('acti_summary.txt', 'w') as file:
    file.write(f"People count: {people_count}\n")
    file.write("Objects in motion:\n")
    for obj in objects_in_motion:
        file.write(f"{obj}\n")

# Print statistics
print("Total frames: ", c)
print("People count: ", people_count)
print("Objects in motion: ", objects_in_motion)
