import os

IMAGES_DIR = "C:/Users/thiba/PycharmProjects/mission_IA_JCS/files/images/"
MASKS_DIR = "C:/Users/thiba/PycharmProjects/mission_IA_JCS/files/labels_masks/"
IMAGE_SIZE = (160, 160)
N_CLASSES = 3
BATCH_SIZE = 32


def get_files_paths(files_dir: str):
    """Get the paths of the files in a given folder."""
    file_paths = sorted(
        [
            os.path.join(files_dir, file_name)
            for file_name in os.listdir(files_dir)
        ]
    )
    return file_paths


image_paths = get_files_paths(IMAGES_DIR)
mask_paths = get_files_paths(MASKS_DIR)

print("Number of images:", len(get_files_paths(IMAGES_DIR)))

for input_path, target_path in zip(image_paths, mask_paths):
    print(input_path, "|", target_path)
