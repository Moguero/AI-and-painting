import urllib.request
import json
import uuid
from loguru import logger
from pathlib import Path

# Example variables
MASK_URL = "https://api.labelbox.com/masks/feature/ckph5r33g00043a6dklihalmq?token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VySWQiOiJja3BneTBhZDc4OXAwMHk5dzZlcWM2bzNlIiwib3JnYW5pemF0aW9uSWQiOiJja3BneTBhY3U4OW96MHk5dzNrcW43MGxmIiwiaWF0IjoxNjIyNzQwNjczLCJleHAiOjE2MjUzMzI2NzN9.VeR0ot2_MAkY769kcXSz8RWqRguopgO1rlbRIGwZWV0"
OUTPUT_PATH = Path("C:/Users/thiba/OneDrive/Documents/Césure/test/test.png")
OUTPUT_DIR_PATH = Path("C:/Users/thiba/PycharmProjects/mission_IA_JCS/files/labels_masks")
JSON_PATH = Path("C:/Users/thiba/OneDrive - CentraleSupelec/Mission_JCS_IA_peinture/labelbox_export_json/export-2021-07-13T17_55_24.684Z.json")


# todo : use try/except statement with urlretrieve to check if the image is well downloaded
# todo : check if we can put the mask_url as a Path
def download_mask(mask_url: str, output_path: Path) -> None:
    """Download the label mask from Labelbox online archive to local machine."""
    # todo : cases where finished by JPG or PNG or ...
    # assert (
    #     output_path[-3:] == "png"
    # ), f"""The filename output path should finish by ".png". \n However, the current output path is {output_path}"""
    assert mask_url[:5] == f"https", f"""The mask url provided is not secured \n It should start by "https". However, the current url is {mask_url}"""

    urllib.request.urlretrieve(url=mask_url, filename=str(output_path))


# todo : check if the Path are correct
# todo : gérer le cas où il a deux masques pour la même catégorie : les fusionner
# todo : put logger with loguru
def download_all_masks(json_path: str, output_dir_path: Path) -> None:
    """Download all the masks"""
    urls_dict = get_mask_urls(json_path)
    for image_name, url_dict in urls_dict.items():
        image_dir_name = output_dir_path / image_name
        if not image_dir_name.exists():
            image_dir_name.mkdir()
        if url_dict:
            for hash_class_name, mask_url in url_dict.items():
                real_class_name = hash_class_name.split('__')[0]
                class_dir_name = image_dir_name / real_class_name
                if not class_dir_name.exists():
                    class_dir_name.mkdir()
                file_name = "mask_" + image_name + "_" + hash_class_name + ".png"
                output_path = class_dir_name / file_name
                breakpoint()
                if output_path.exists():
                    logger.warning(f"\nFile {file_name} already exists : overwrite previous file")
                download_mask(mask_url=mask_url, output_path=output_path)


# todo : retyper l'output en détaillant le dict
def get_mask_urls(json_path: str) -> dict:
    """Associate an image name with a list of its corresponding class mask URLs"""
    mask_urls = dict()
    with open(json_path) as file:
        json_list = json.load(file)
        for image_dict in json_list:
            reformatted_external_id = image_dict["External ID"].split(".")[0]
            mask_urls[reformatted_external_id] = dict()
            if bool(image_dict["Label"]):
                image_objects = image_dict["Label"]["objects"]
                for image_object in image_objects:
                    mask_url = image_object["instanceURI"]
                    mask_urls[reformatted_external_id][image_object["title"] + '__' + uuid.uuid4().hex] = mask_url
                logger.info(
                    f"""\nNumber of masks for image with external id "{reformatted_external_id}" : {len(mask_urls[reformatted_external_id])}"""
                )
            else:
                logger.warning(
                    f"\nNo masks for image with external id : {reformatted_external_id}"
                )
    return mask_urls


# DEBUG

# todo : retyper l'output
def get_full_json(json_path: str) -> list:
    with open(json_path) as file:
        json_dict = json.load(file)
    return json_dict
