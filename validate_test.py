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


def preprocess_image(image_path, input_shape):
    # Load image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise Exception("Image not loaded properly.")

    img_resized = cv2.resize(img, (28, 28))  # shape is (width, height)

    # Save the resized image to a file in JPEG format
    resized_image_path = "resized_for_network.jpg"
    cv2.imwrite(resized_image_path, img_resized)
    print(f"Resized image saved to {resized_image_path}")

    # Resize image to match the neural network's expected input dimensions
    img_reshape = img_resized.reshape(input_shape)

    # Normalize the image (0-255 to 0-1 scaling)
    # img_normalized = img_resized / 255.0
    img_resized_uint8 = img_reshape.astype(np.uint8)
    # img_normalized_uint8 = (img_normalized * 255).astype(np.uint8)

    # Add batch dimension and reshape to input shape
    # img_reshaped = img_normalized.reshape(input_shape)
    img_reshaped = img_resized_uint8.reshape(input_shape)

    return img_reshaped


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
    input_shape = driver.ibuf_packed_device[0].shape  # Expected input shape
    preprocessed_image = preprocess_image(image_path, input_shape)

    # Copy preprocessed image data to the device
    driver.copy_input_data_to_device(preprocessed_image)

    # Run inference
    driver.execute_on_buffers()

    # Retrieve the output
    obuf_normal = np.empty_like(driver.obuf_packed_device[0])
    driver.copy_output_data_from_device(obuf_normal)

    # Process the output (this part depends on your specific network and application)
    # For example, assuming output is a classification result:
    predicted_label = obuf_normal.flatten()
    print("Predicted label: ", predicted_label)

