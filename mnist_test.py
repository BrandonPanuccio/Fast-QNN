import argparse
import numpy as np
import cv2  # Import OpenCV to save image as JPEG
from driver import io_shape_dict
from driver_base import FINNExampleOverlay
from dataset_loading import mnist

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Use a single sample image from the MNIST dataset as input for the FINN accelerator"
    )
    parser.add_argument(
        "--bitfile", help='name of bitfile (i.e. "resizer.bit")', default="resizer.bit"
    )
    parser.add_argument(
        "--platform", help="Target platform: zynq-iodma or alveo", default="zynq-iodma"
    )
    parser.add_argument(
        "--dataset_root", help="dataset root dir for download/reuse", default="/tmp"
    )
    args = parser.parse_args()
    bitfile = args.bitfile
    platform = args.platform
    dataset_root = args.dataset_root

    # Load the MNIST dataset
    trainx, trainy, testx, testy, valx, valy = mnist.load_mnist_data(
        dataset_root, download=True, one_hot=False
    )

    # Select a random sample image and label
    random_index = np.random.choice(len(testx))  # Randomly select an index
    sample_image = testx[random_index]
    sample_label = testy[random_index]

    # Convert sample image to uint8 for saving as JPEG
    sample_image_uint8 = (sample_image * 255).astype(np.uint8)  # Scale and convert to uint8
    cv2.imwrite("mnist_sample_image.jpg", sample_image_uint8)

    # Set up the FINNExampleOverlay
    driver = FINNExampleOverlay(
        bitfile_name=bitfile,
        platform=platform,
        io_shape_dict=io_shape_dict,
        batch_size=1,  # Using only one sample
        runtime_weight_dir="runtime_weights/",
    )

    # Prepare and load input into the driver
    ibuf_normal = sample_image.reshape(driver.ibuf_packed_device[0].shape)
    driver.copy_input_data_to_device(ibuf_normal)

    # Run inference on the single sample image
    driver.execute_on_buffers()

    # Retrieve the output from the device
    obuf_normal = np.empty_like(driver.obuf_packed_device[0])
    driver.copy_output_data_from_device(obuf_normal)

    # Get the predicted label by finding the index of the maximum value in the output
    predicted_label = obuf_normal.flatten()

    # Print the predicted label
    print("Predicted label by the model:", predicted_label)
    print("The correct label:", sample_label)
