from flask import Flask, render_template, jsonify
import threading
import cv2
import numpy as np
import time
from datetime import datetime
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

app = Flask(__name__)

# Flag to indicate whether summarization is in progress
summarization_in_progress = False

# Thread to run the video summarization system
def run_summarization():
    global summarization_in_progress

    # Open a connection to the default camera (index 0)
    video = cv2.VideoCapture(1)

    # Get video properties
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create VideoWriters for both original and summarized videos
    original_writer = cv2.VideoWriter('orig_video.mp4', cv2.VideoWriter_fourcc(*'DIVX'), 25, (width, height))
    summarized_writer = cv2.VideoWriter('summarized_video.mp4', cv2.VideoWriter_fourcc(*'DIVX'), 25, (width, height))

    # Set the duration of webcam activation (90 seconds)
    duration = 90  # seconds
    end_time = time.time() + duration

    # Create VideoWriter for recording events
    event_writer = None

    # Virtual fence parameters
    fence_line_x = int(0.2 * width)  # X-coordinate for the virtual fence line
    fence_color = (0, 255, 0)  # Green color for the fence line
    fence_thickness = 2  # Thickness of the fence line

    # Email configuration (replace with your own details)
    smtp_server = "smtp.gmail.com"
    smtp_port = 587  # Adjust if needed
    smtp_username = "machariaalex456@gmail.com"
    smtp_password = "htlm qbby meqi sqwy"
    recipient_email = "cheronotruphena66@gmail.com"  # Replace with the recipient's email address

    def send_email(subject, body):
        # Setup the MIME
        message = MIMEMultipart()
        message['From'] = smtp_username
        message['To'] = recipient_email
        message['Subject'] = subject

        # Attach the message body
        message.attach(MIMEText(body, 'plain'))

        # Connect to the SMTP server
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            # Login to the email account
            server.starttls()  # Use if your server requires a secure connection
            server.login(smtp_username, smtp_password)

            # Send the email
            server.sendmail(smtp_username, recipient_email, message.as_string())

    # Initialize variables
    ret, frame1 = video.read()
    prev_frame = frame1
    a = 0
    b = 0
    c = 0
    object_count = 0
    people_count = 0
    objects_in_motion = set()

    # Load MobileNet SSD model
    protopath = 'MobileNetSSD_deploy.prototxt'
    modelpath = 'MobileNetSSD_deploy.caffemodel'
    detector = cv2.dnn.readNetFromCaffe(prototxt=protopath, caffeModel=modelpath)

    # Define object classes
    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
               "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
               "sofa", "train", "tvmonitor"]

    # List to store frame paths
    frames_path = []

    # Main loop
    while time.time() < end_time:
        # Read a frame from the camera
        ret, frame = video.read()

        # Perform object detection
        (H, W) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
        detector.setInput(blob)
        person_detections = detector.forward()

        # Draw virtual fence line
        cv2.line(frame, (fence_line_x, 0), (fence_line_x, height), fence_color, fence_thickness)

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
                    event_writer = cv2.VideoWriter(f'event_{time.time()}.mp4', cv2.VideoWriter_fourcc(*'DIVX'), 25,
                                                  (width, height))

                event_writer.write(frame)

                # Save the frame as an image
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                frame_filename = f'static/frames/frame_{timestamp}.png'  # Save frames in 'static/frames' folder
                cv2.imwrite(frame_filename, frame)
                frames_path.append(frame_filename)

                # Check if the person/object crossed the virtual fence
                if startX < fence_line_x < endX:
                    # Save an image of the trespasser as .png
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    image_filename = f'static/trespassers/trespasser_{timestamp}.png'  # Save trespassers in 'static/trespassers' folder
                    cv2.imwrite(image_filename, frame)

                    # Notify about the trespasser
                    send_email("Trespasser Alert", f"Someone has trespassed your premise. Image attached: {image_filename}")

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
    with open('activity_summary.txt', 'w') as file:
        file.write(f"People count: {people_count}\n")
        file.write("Objects in motion:\n")
        for obj in objects_in_motion:
            file.write(f"{obj}\n")

    # Print statistics
    print("Total frames: ", c)
    print("People count: ", people_count)
    print("Objects in motion: ", objects_in_motion)

    # Set summarization_in_progress to False
    summarization_in_progress = False

# Route to render the main page
@app.route('/')
def index():
    return render_template('web.html', summarization_in_progress=summarization_in_progress)

# Route to start video summarization
@app.route('/start_summarization')
def start_summarization():
    global summarization_in_progress

    # Check if summarization is not already in progress
    if not summarization_in_progress:
        # Start a new thread for video summarization
        summarization_thread = threading.Thread(target=run_summarization)
        summarization_thread.start()

        # Set summarization_in_progress to True
        summarization_in_progress = True

        return jsonify({'status': 'success'})
    else:
        return jsonify({'status': 'error', 'message': 'Summarization already in progress'})

if __name__ == '__main__':
    app.run(debug=True)
