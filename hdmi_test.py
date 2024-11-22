import cv2
from pynq.overlays.base import BaseOverlay
from pynq.lib.video import *

# Load the base overlay with HDMI support
base = BaseOverlay("base.bit")
hdmi_out = base.video.hdmi_out

# Configure the HDMI output with a standard mode (e.g., 640x480)
hdmi_out.configure(VideoMode(640, 480, 24), PIXEL_RGB)
hdmi_out.start()

# Open the USB webcam
cap = cv2.VideoCapture('/dev/video0', cv2.CAP_V4L2)
if not cap.isOpened():
    print("Error: Could not open webcam.")
else:
    try:
        while True:
            # Capture a frame from the webcam
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame.")
                break

            # Resize the frame to match the HDMI output resolution
            resized_frame = cv2.resize(frame, (640, 480))

            # Create a new VideoFrame for HDMI output
            video_frame = hdmi_out.newframe()
            video_frame[0:480, 0:640, :] = resized_frame[:, :, :]

            # Send the frame to HDMI output
            hdmi_out.writeframe(video_frame)

            # Optional: Add a small delay to reduce CPU load
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("Program interrupted by user.")

    finally:
        # Release resources
        cap.release()
        hdmi_out.stop()
        del hdmi_out
