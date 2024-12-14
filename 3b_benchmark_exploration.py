import os.path
import urllib.request
import zipfile

import cv2
import numpy as np
from PIL import Image

zip_url = "https://data.broadinstitute.org/bbbc/BBBC031/BBBC031_v1_dataset.zip"
target_path = "data/BBBC031_v1_dataset"

if not os.path.exists(target_path):
    # download the zip file
    urllib.request.urlretrieve(zip_url, f"{target_path}.zip")

    # unpack the zip file
    with zipfile.ZipFile(f"{target_path}.zip", 'r') as zip_ref:
        zip_ref.extractall(target_path)


# read the images from the dataset
image_paths = os.listdir(f"{target_path}/Images")

for image_path in image_paths:
    print(image_path)

    # open the image (png) file
    image = Image.open(f"{target_path}/Images/{image_path}")

    # get the nuclius mask (tiff) file name
    label_name = image_path.replace(".png", ".tiff")
    label_name = label_name.replace("CELLMASK", "NUCLMASK")
    label_path = f"{target_path}/Masks/{label_name}"

    # read the label (tiff) image
    label = Image.open(label_path)

    # show the images and labels
    image.show()
    label.show()

    # count the number of nuclei in the label (number of connected commponents)
    num_nuclei = cv2.connectedComponents(np.array(label))[0]
    print(f"Number of nuclei: {num_nuclei}")

    break
