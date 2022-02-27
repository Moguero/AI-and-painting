import argparse
import time
import json
import uuid
import urllib.request
from loguru import logger
from pathlib import Path


def get_masks_urls(json_path: Path) -> dict:
    """
    Associate an images name with a list of its corresponding class mask URLs.

    :param json_path : The JSON exported from LabelBox.
    :return: A dictionary of tuples (<image_name>, <dictionary_class_name/mask_url>)
    """
    mask_urls = dict()
    with open(str(json_path)) as file:
        json_list = json.load(file)
        for image_dict in json_list:
            reformatted_external_id = image_dict["External ID"].split(".")[0]
            mask_urls[reformatted_external_id] = dict()
            if bool(image_dict["Label"]):
                image_objects = image_dict["Label"]["objects"]
                for image_object in image_objects:
                    mask_url = image_object["instanceURI"]
                    mask_urls[reformatted_external_id][
                        image_object["title"] + "__" + uuid.uuid4().hex
                    ] = mask_url
                logger.info(
                    f"""\nNumber of masks for images with external id "{reformatted_external_id}" : {len(mask_urls[reformatted_external_id])}"""
                )
            else:
                logger.warning(
                    f"\nNo masks for images with external id : {reformatted_external_id}"
                )
    return mask_urls


def count_masks_urls(mask_urls: dict) -> int:
    """

    :param mask_urls: The dictionary returned by get_mask_urls function.
    :return: The total number of masks URLs in the JSON.
    """
    return sum([len(item[1]) for item in mask_urls.items()])


def download_mask(mask_url: str, output_path: Path) -> None:
    """
    Download a binary label mask from Labelbox online archive to the output path specified.

    :param mask_url : The LabelBox online archive URL of the mask we want to download.
    :param output_path : The path where to store the downloaded mask.
    """
    filename = output_path.parts[-1]
    assert (
        filename[-3:] == "png"
    ), f"""The filename output path should finish by ".png". \n However, the current output path is {output_path}"""
    assert (
        mask_url[:5] == f"https"
    ), f"""The mask url provided is not secured \n It should start by "https". However, the current url is {mask_url}"""

    urllib.request.urlretrieve(url=mask_url, filename=str(output_path))
    logger.info(f"\nMask {filename} downloaded successfully.")


def download_all_masks(json_path: Path, output_dir_path: Path) -> None:
    """
    Download all the labels masks.

    :param json_path : The JSON exported from LabelBox.
    :param output_dir_path: The directory where we want to save the masks.
    """
    urls_dict = get_masks_urls(json_path)
    number_of_masks = count_masks_urls(urls_dict)
    logger.info(f"\nStarting to download {number_of_masks} masks...")
    for image_name, url_dict in urls_dict.items():
        image_dir_name = output_dir_path / image_name
        if not image_dir_name.exists():
            image_dir_name.mkdir()
            logger.info(f"\nImage folder {image_dir_name} was created.")
        if url_dict:
            for hash_class_name, mask_url in url_dict.items():
                real_class_name = hash_class_name.split("__")[0]
                class_dir_name = image_dir_name / real_class_name
                if not class_dir_name.exists():
                    class_dir_name.mkdir()
                    logger.info(f"\nClass folder {class_dir_name} was created.")
                file_name = "mask_" + image_name + "_" + hash_class_name + ".png"
                output_path = class_dir_name / file_name
                if output_path.exists():
                    logger.warning(
                        f"\nFile {file_name} already exists : overwrite previous file"
                    )
                download_mask(mask_url=mask_url, output_path=output_path)
    logger.info(f"\n{number_of_masks} masks downloaded successfully")


# Creation of the CLI command
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download labels masks.")
    parser.add_argument(
        "--json-path", type=str, help="The JSON exported from LabelBox.", required=True
    )
    parser.add_argument(
        "--output-dir-path",
        type=str,
        help="The directory where we want to save the masks.",
        required=True,
    )
    args = parser.parse_args()
    return args


def main() -> None:
    logger.info("\nStarting to download labels masks from LabelBox...")
    start_time = time.time()
    args = parse_args()
    json_path = Path(args.json_path)
    output_dir_path = Path(args.output_dir_path)
    download_all_masks(json_path, output_dir_path)
    logger.info(
        f"\nMasks downloading finished in {(time.time() - start_time)/60:.1f} minutes.\n"
    )


if __name__ == "__main__":
    main()


# -----------------
# DEBUG
def get_full_json(json_path: Path) -> [dict]:
    with open(str(json_path)) as file:
        json_dict = json.load(file)
    return json_dict
