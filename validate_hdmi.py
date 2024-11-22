from PIL import Image
import argparse
import cv2
import numpy as np
import time
from driver import io_shape_dict
from driver_base import FINNExampleOverlay

from pynq.overlays.base import BaseOverlay
from pynq.lib.video import *


def predict_image(preprocessed_image, driver):
    ibuf_normal = preprocessed_image.reshape(driver.ibuf_packed_device[0].shape)

    # Perform inference with FINN
    driver.copy_input_data_to_device(ibuf_normal)
    driver.execute_on_buffers()
    obuf_normal = np.empty_like(driver.obuf_packed_device[0])
    driver.copy_output_data_from_device(obuf_normal)
    predicted_label = obuf_normal.flatten()

    # Print the detected number
    print("Predicted label by the model:", predicted_label)


def preprocess_image(image):
    # Convert image to grayscale if needed
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Invert colors so that digits appear white on black background
    img_inverted = 255 - gray

    # Apply binary thresholding to distinguish number
    _, img_binary = cv2.threshold(img_inverted, 128, 255, cv2.THRESH_BINARY)

    # Find contours to identify and isolate the digit area
    contours, _ = cv2.findContours(img_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        img_digit = img_binary[y:y + h, x:x + w]

        # Resize the cropped digit to a 20x20 box
        img_resized_digit = cv2.resize(img_digit, (20, 20), interpolation=cv2.INTER_AREA)
    else:
        img_resized_digit = cv2.resize(img_binary, (20, 20), interpolation=cv2.INTER_AREA)

    # Create a blank 28x28 black canvas and place the 20x20 digit in the center
    img_mnist_style = np.zeros((28, 28), dtype=np.uint8)
    img_mnist_style[4:24, 4:24] = img_resized_digit  # Center the 20x20 digit in a 28x28 canvas

    # Return the 28x28 image
    img_mnist_style = img_mnist_style.reshape(1, 28, 28)

    return img_mnist_style


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Perform inference on a captured webcam frame using FINN-generated accelerator"
    )
    parser.add_argument(
        "--platform", help="Target platform: zynq-iodma alveo", default="zynq-iodma"
    )
    parser.add_argument(
        "--bitfile", help='name of bitfile (i.e. "resizer.bit")', default="resizer.bit"
    )
    # Parse arguments
    args = parser.parse_args()
    bitfile = args.bitfile
    platform = args.platform

    # Load the base overlay with HDMI and FINN capability
    base = BaseOverlay("base.bit")
    hdmi_out = base.video.hdmi_out

    # Configure HDMI output
    hdmi_out.configure(VideoMode(640, 480, 24), PIXEL_RGB)
    hdmi_out.start()

    # Load the FINN driver once, assuming it's compatible with the base overlay
    finn_driver = FINNExampleOverlay(
        bitfile_name=bitfile,
        platform=platform,
        io_shape_dict=io_shape_dict,
        batch_size=1,
        runtime_weight_dir="runtime_weights/"
    )

    # Open webcam
    cap = cv2.VideoCapture('/dev/video0', cv2.CAP_V4L2)
    if not cap.isOpened():
        print("Could not open webcam.")
    else:
        try:
            # Wait for a non-black frame before starting
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Could not read frame.")
                    break
                if np.mean(frame) > 10:  # Adjust threshold if necessary
                    break

            counter = 0
            while True:  # Run indefinitely until interrupted
                # Capture frame
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

                # Process frame every 10 frames
                if counter == 10:
                    # Freeze HDMI output on the current frame
                    print("Freezing HDMI output for processing...")
                    hdmi_out.writeframe(video_frame)

                    # Countdown before processing
                    for i in range(3, 0, -1):
                        print(f"Processing will start in {i} seconds...")
                        time.sleep(1)

                    # Preprocess the captured frame
                    preprocessed_image = preprocess_image(frame)

                    # Perform prediction with the pre-loaded FINN driver
                    predict_image(preprocessed_image, finn_driver)

                    # Reset counter
                    counter = 0
                else:
                    counter += 1

                # Optional: Add a small delay to reduce CPU load
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except KeyboardInterrupt:
            print("\nProgram interrupted by user.")

        finally:
            # Release resources
            cap.release()
            hdmi_out.stop()
            del hdmi_out
