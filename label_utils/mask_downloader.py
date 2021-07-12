import urllib.request
import argparse
import os

from label_utils.labelbox_json_mask_urls_extractor import get_mask_urls
from loguru import logger

# Example variables
mask_url = "https://api.labelbox.com/masks/feature/ckph5r33g00043a6dklihalmq?token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VySWQiOiJja3BneTBhZDc4OXAwMHk5dzZlcWM2bzNlIiwib3JnYW5pemF0aW9uSWQiOiJja3BneTBhY3U4OW96MHk5dzNrcW43MGxmIiwiaWF0IjoxNjIyNzQwNjczLCJleHAiOjE2MjUzMzI2NzN9.VeR0ot2_MAkY769kcXSz8RWqRguopgO1rlbRIGwZWV0"
output_path = "C:/Users/thiba/OneDrive/Documents/Césure/test/test.png"
output_dir = "C:/Users/thiba/OneDrive - CentraleSupelec/Mission_JCS_IA_peinture/masks/test/"
output_dir = "C:/Users/thiba/PycharmProjects/mission_IA_JCS/files/labels_masks/"
json_path = "C:/Users/thiba/OneDrive - CentraleSupelec/Mission_JCS_IA_peinture/labelbox_export_json/export-2021-07-01T16_53_01.845Z.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download an image from an url.")
    parser.add_argument(
        "--url", type=str, help="The mask url from Labelbox.", required=True
    )
    parser.add_argument(
        "--output-path",
        type=str,
        help="The path where the retrieved image will be stored.",
        required=True,
    )
    args = parser.parse_args()
    return args


def download_mask(mask_url: str, output_path: str) -> None:
    """Download the label mask from Labelbox online archive to local machine."""
    # todo : cases where finished by JPG or PNG or ...
    # assert (
    #     output_path[-3:] == "png"
    # ), f"""The filename output path should finish by ".png". \n However, the current output path is {output_path}"""
    assert mask_url[:5] == f"https", f"""The mask url provided is not secured \n It should start by "https". However, the current url is {mask_url}"""

    urllib.request.urlretrieve(url=mask_url, filename=output_path)


# todo : download the masks only if the
# todo : put logger with loguru
def download_all_masks(json_path: str, output_dir: str) -> None:
    """Download all the masks"""
    urls_dict = get_mask_urls(json_path)
    for image_name, url_list in urls_dict.items():
        if url_list:
            for idx, url_dict in enumerate(url_list):
                for class_name, mask_url in url_dict.items():
                    # todo : create folder with image_name
                    # todo : take into account the case where directory already exists
                    # todo : check that the images are well downloaded
                    # https://thispointer.com/how-to-create-a-directory-in-python/#:~:text=Python's%20OS%20module%20provides%20an%20another%20function%20to%20create%20a%20directories%20i.e.&text=.makedirs(path)-,os.,mkdir%20%2Dp%20command%20in%20linux.
                    sub_dir = output_dir + image_name + '/'
                    file_name = "mask_" + image_name + "_" + class_name + ".png"
                    output_path = sub_dir + file_name
                    if not os.path.exists(sub_dir):
                        os.mkdir(sub_dir)
                    if os.path.exists(output_path):
                        logger.warning(f"\nFile {file_name} already exists : overwrite previous file")
                    download_mask(mask_url=mask_url, output_path=output_path)


def main() -> None:
    args = parse_args()
    download_mask(args.url, args.output_path)


# if __name__ == "__main__":
#     main()
