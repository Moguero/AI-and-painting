import json
from loguru import logger

json_path = "C:/Users/thiba/OneDrive - CentraleSupelec/Mission_JCS_IA_peinture/labelbox_export_json/export-2021-07-12T14_48_36.100Z.json"


def get_mask_urls(json_path: str) -> dict:
    """Associate an image name with a list of its corresponding class mask URLs"""
    mask_urls = dict()
    with open(json_path) as file:
        json_dict = json.load(file)
        for image in json_dict:
            reformatted_external_id = image["External ID"].split(".")[0]
            mask_urls[reformatted_external_id] = list()
            if bool(image["Label"]):
                image_objects = image["Label"]["objects"]
                for image_object in image_objects:
                    mask_url = image_object["instanceURI"]
                    mask_urls[reformatted_external_id].append({image_object["title"]: mask_url})
                logger.info(f"""\nNumber of masks for image with external id "{reformatted_external_id}" : {len(mask_urls)} """)
            else:
                logger.warning(f"\nNo masks for image with external id : {reformatted_external_id}")
    return mask_urls


# todo : delete this tests because several masks when several label classes
def test_check_one_mask_per_image(json_path: str):
    """Checks if there is maximum one mask per image"""
    number_of_masks_per_image = dict()
    with open(json_path) as f:
        json_dict = json.load(f)
        for image in json_dict:
            external_id = image["External ID"]
            if bool(image["Label"]):
                image_objects = image["Label"]["objects"]
                number_of_masks_per_image[external_id] = len(image_objects)
            else:
                number_of_masks_per_image[external_id] = 0
    assert 0 not in number_of_masks_per_image.values()
    # TODO : return an error message with the name of the concerned image with no mask
