from PIL import Image
import cv2
import argparse
import numpy as np
from driver import io_shape_dict
from driver_base import FINNExampleOverlay

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Validate top-1 accuracy for FINN-generated accelerator"
    )
    parser.add_argument(
        "--batchsize", help="number of samples for inference", type=int, default=1
    )

    parser.add_argument(
        "--platform", help="Target platform: zynq-iodma alveo", default="zynq-iodma"
    )
    parser.add_argument(
        "--bitfile", help='name of bitfile (i.e. "resizer.bit")', default="resizer.bit"
    )
    # parse arguments
    args = parser.parse_args()
    bsize = args.batchsize
    bitfile = args.bitfile
    platform = args.platform

    driver = FINNExampleOverlay(
        bitfile_name=bitfile,
        platform=platform,
        io_shape_dict=io_shape_dict,
        batch_size=bsize,
        runtime_weight_dir="runtime_weights/",
    )
    img = Image.open("sample3.jpg").convert('L')
    img_np = np.asarray(img)
    if np.mean(img_np) > 127:
        img_invert = 255 - img_np
        img_resized = cv2.resize(img_invert, (28, 28))  # shape is (width, height)
    else:
        img_resized = cv2.resize(img_np, (28, 28))

    # Save the resized image to a file in JPEG format
    resized_image_path = "sample_resized.jpg"
    cv2.imwrite(resized_image_path, img_resized)
    print(f"Resized image saved to {resized_image_path}")

    img_resized.reshape(1, 28, 28).transpose(1, 2, 0)

    ibuf_normal = img_resized.reshape(driver.ibuf_packed_device[0].shape)
    driver.copy_input_data_to_device(ibuf_normal)
    driver.execute_on_buffers()
    obuf_normal = np.empty_like(driver.obuf_packed_device[0])
    driver.copy_output_data_from_device(obuf_normal)
    predicted_label = obuf_normal.flatten()

    # Print the predicted label
    print("Predicted label by the model:", predicted_label)

