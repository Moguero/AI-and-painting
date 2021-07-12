import os
from shutil import copyfile
from loguru import logger

IMAGE_SOURCE_PATH = "C:/Users/thiba/OneDrive - CentraleSupelec/Mission_JCS_IA_peinture/images/AVZD6310.JPG"
IMAGE_TARGET_DIR_PATH = "C:/Users/thiba/PycharmProjects/mission_IA_JCS/files/images/"


def move_image(image_source_path: str, image_target_dir_path: str):
    image_name = image_source_path.split("/")[-1].split(".")[0]
    sub_dir = image_target_dir_path + image_name + '/'
    file_name = image_name + ".png"
    output_path = sub_dir + file_name
    if not os.path.exists(sub_dir):
        os.mkdir(sub_dir)
        logger.info(f"Sub folder {sub_dir} was created")
    copyfile(image_source_path, output_path)
