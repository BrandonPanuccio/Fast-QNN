from PIL import Image
import cv2
import numpy as np
from flask import Flask, Response
import argparse
import time

from driver import io_shape_dict
from driver_base import FINNExampleOverlay

app = Flask(__name__)  # Flask app for streaming video


def preprocess_image(image):
    # Load image and convert to grayscale
    # img = Image.open(image).convert('L')
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Convert image to numpy array
    img_np = np.array(img)

    # Invert colors so that digits appear white on black background
    img_inverted = 255 - img_np

    # Apply binary thresholding to distinguish number
    # values above 128 will be set to 255
    _, img_binary = cv2.threshold(img_inverted, 128, 255, cv2.THRESH_BINARY)

    # Find contours to identify and isolate the digit area
    contours, _ = cv2.findContours(img_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)  # Get bounding box of the largest contour
        img_digit = img_binary[y:y + h, x:x + w]  # Crop to the bounding box

        # Resize the cropped digit to a 20x20 box (MNIST digits are centered within a 28x28 frame)
        img_resized_digit = cv2.resize(img_digit, (20, 20), interpolation=cv2.INTER_AREA)
    else:
        # If no contour is found, assume the entire area is the digit
        img_resized_digit = cv2.resize(img_binary, (20, 20), interpolation=cv2.INTER_AREA)

    # Create a blank 28x28 black canvas and place the 20x20 digit in the center
    img_mnist_style = np.zeros((28, 28), dtype=np.uint8)
    img_mnist_style[4:24, 4:24] = img_resized_digit  # Center the 20x20 digit in a 28x28 canvas

    # Return the 28x28 image, reshaped for neural network input if needed
    img_mnist_style = img_mnist_style.reshape(1, 28, 28)  # .transpose(1, 2, 0)

    return img_mnist_style


def make_prediction(image, driver):
    """Use the FINN accelerator to make a prediction."""
    ibuf_normal = image.reshape(driver.ibuf_packed_device[0].shape)
    driver.copy_input_data_to_device(ibuf_normal)
    driver.execute_on_buffers()
    obuf_normal = np.empty_like(driver.obuf_packed_device[0])
    driver.copy_output_data_from_device(obuf_normal)
    predicted_label = obuf_normal.flatten()
    return predicted_label


def generate_frames(driver, prediction_interval=2):
    """Generator to capture, process, and stream video frames."""
    cap = cv2.VideoCapture('/dev/video0', cv2.CAP_V4L2)
    if not cap.isOpened():
        raise Exception("Could not open webcam.")

    last_prediction_time = time.time()
    prediction = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Make a prediction periodically
            if time.time() - last_prediction_time > prediction_interval:
                # Preprocess the frame and make a prediction
                preprocessed_image = preprocess_image(frame)
                prediction = make_prediction(preprocessed_image, driver)
                last_prediction_time = time.time()

            # Overlay the prediction on the video frame
            if prediction is not None:
                cv2.putText(
                    frame,
                    f"Prediction: {prediction[0]}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

            # Encode the frame as JPEG
            _, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    finally:
        cap.release()


@app.route('/video_feed')
def video_feed():
    """Route for live video feed with predictions."""
    return Response(generate_frames(driver), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stream video with periodic predictions from a USB camera."
    )
    parser.add_argument(
        "--platform", help="Target platform: zynq-iodma alveo", default="zynq-iodma"
    )
    parser.add_argument(
        "--bitfile", help='Name of bitfile (e.g., "resizer.bit")', default="resizer.bit"
    )
    parser.add_argument(
        "--interval", type=int, help="Prediction interval in seconds", default=2
    )
    args = parser.parse_args()

    # Load the FINN accelerator driver
    driver = FINNExampleOverlay(
        bitfile_name=args.bitfile,
        platform=args.platform,
        io_shape_dict=io_shape_dict,
        batch_size=1,
        runtime_weight_dir="runtime_weights/"
    )

    # Start Flask server for video streaming with predictions
    print("Starting video streaming server with predictions...")
    app.run(host='0.0.0.0', port=5000)

    # run on http://192.168.2.99:5000/video_feed
