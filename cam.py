import cv2
import numpy as np
import smtplib
from email.mime.text import MIMEText

# Function to send email
def send_email():
    sender_email = "machariaalex456@gmail.com"  # Replace with your Gmail email address
    sender_password = "htlm qbby meqi sqwy"  # Replace with your App Password

    subject = "Trespass Alert"
    body = "Someone has trespassed the premises!"

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = sender_email
    msg["To"] = "machariaalex459@gmail.com"  # Replace with the recipient's email address

    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        server.starttls()
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, "machariaalex459@gmail.com", msg.as_string())

# Rest of the code...

video = cv2.VideoCapture(0)

# Get video properties
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

original_writer = cv2.VideoWriter('original_video.mp4', cv2.VideoWriter_fourcc(*'XVID'), 25, (width, height))
summarized_writer = cv2.VideoWriter('summarized_video.mp4', cv2.VideoWriter_fourcc(*'XVID'), 25, (width, height))


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

critical_area_x = width - 20  # Adjust the position of the critical area line
critical_area_color = (0, 255, 0)  # Green color

while True:
    ret, frame = video.read()

    # Add a green virtual fence line slightly to the right
    cv2.line(frame, (critical_area_x, 0), (critical_area_x, height), critical_area_color, 2)

    # Object detection using MobileNet SSD
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
    detector.setInput(blob)
    detections = detector.forward()

    # Process the detections
    person_count = 0

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:  # Confidence threshold
            class_id = int(detections[0, 0, i, 1])

            if CLASSES[class_id] == "person":
                person_count += 1
                x, y, w, h = (int(detections[0, 0, i, 3] * width), int(detections[0, 0, i, 4] * height),
                              int((detections[0, 0, i, 5] - detections[0, 0, i, 3]) * width),
                              int((detections[0, 0, i, 6] - detections[0, 0, i, 4]) * height))

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Check if the person crosses the green line
                if x + w > critical_area_x:
                    send_email()

    # Update object count
    object_count += person_count

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
