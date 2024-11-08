import cv2
import matplotlib.pyplot as plt
from IPython.display import display, clear_output


# Initialize the webcam (0 is typically the default camera)
cap = cv2.VideoCapture('/dev/video0', cv2.CAP_V4L2)
#cap = cv2.VideoCapture("V4l2src device=/dev/video0 ! videoconvert ! appsink", cv2.CAP_GSTREAMER)
print("Connected")

if not cap.isOpened():
    print("Error: Could not open webcam.")
    cap.release()
else:
    try:
        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame.")
                break

            # Convert the frame color from BGR to RGB for matplotlib
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Display the frame in Jupyter Notebook
            plt.imshow(frame_rgb)
            plt.axis('off')  # Hide axes for a cleaner look

            # Update the display
            clear_output(wait=True)
            display(plt.gcf())
            plt.pause(0.0001)  # Short pause to allow for real-time display
    except KeyboardInterrupt:
        # Stop the loop with Ctrl+C or interrupt the kernel
        print("Stopped by user.")

    finally:
        # Release the webcam and close the display
        cap.release()
        clear_output()
        print("Camera released.")