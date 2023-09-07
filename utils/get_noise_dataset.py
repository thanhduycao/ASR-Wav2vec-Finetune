import zipfile
from utils.utils import runcmd
import os


def get_noise_dataset(url_path, output_path):
    runcmd(f"wget -P {output_path} {url_path}", verbose=True)
    # list to store files
    res = []

    # Iterate directory
    for file_path in os.listdir(output_path):
        # check if current file_path is a file
        if os.path.isfile(os.path.join(output_path, file_path)):
            # add filename to list
            res.append(file_path)

    zip_path = os.path.join(output_path, res[0])
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(output_path)
    os.remove(zip_path)
