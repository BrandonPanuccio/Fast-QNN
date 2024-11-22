from PIL import Image
import argparse
import cv2
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
import numpy as np

# Now import the drivers from /root/home
from driver import io_shape_dict
from driver_base import FINNExampleOverlay


def capture_frame(save_path="captured_frame.jpg"):
    # Open webcam
    cap = cv2.VideoCapture('/dev/video0', cv2.CAP_V4L2)

    if not cap.isOpened():
        raise Exception("Could not open webcam.")
        cap.release()
    else:
        try:
            # Wait until we get a non-black frame
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Could not read frame.")
                    break

                # Check if the frame is mostly black by looking at the pixel intensity
                if np.mean(frame) > 10:  # Adjust threshold if necessary
                    break  # Break the loop once we get a non-black frame

            # Capture frame
            ret, frame = cap.read()
            if not ret:
                raise Exception("Failed to capture image from webcam.")

            # Save frame to file
            cv2.imwrite(save_path, frame)

        finally:
            # Release the camera
            cap.release()

        # Return the saved frame path
        return save_path


def preprocess_image(image_path):
    # Load image and convert to grayscale
    img = Image.open(image_path).convert('L')

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

    # Save the processed image for reference
    resized_image_path = "resized_for_network.jpg"
    cv2.imwrite(resized_image_path, img_mnist_style)
    print(f"Processed image saved to {resized_image_path}")

    # Return the 28x28 image, reshaped for neural network input if needed
    img_mnist_style = img_mnist_style.reshape(1, 28, 28)  # .transpose(1, 2, 0)

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
    parser.add_argument(
        "--save_path", help="Path to save the captured frame", default="captured_frame.jpg"
    )
    # parse arguments
    args = parser.parse_args()
    bitfile = args.bitfile
    platform = args.platform
    save_path = args.save_path

    # Capture an image from the webcam and save it
    image_path = capture_frame(save_path)

    # Load the FINN accelerator driver
    driver = FINNExampleOverlay(
        bitfile_name=bitfile,
        platform=platform,
        io_shape_dict=io_shape_dict,
        batch_size=1,  # Since we are processing one image at a time
        runtime_weight_dir="runtime_weights/"
    )

    # Preprocess the captured image
    preprocessed_image = preprocess_image(image_path)
    ibuf_normal = preprocessed_image.reshape(driver.ibuf_packed_device[0].shape)
    driver.copy_input_data_to_device(ibuf_normal)
    driver.execute_on_buffers()
    obuf_normal = np.empty_like(driver.obuf_packed_device[0])
    driver.copy_output_data_from_device(obuf_normal)
    predicted_label = obuf_normal.flatten()

    # Print the predicted label
    print("Predicted label by the model:", predicted_label)

