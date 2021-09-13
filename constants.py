from tensorflow.python.keras.optimizers import Adam
from pathlib import Path
from tensorflow import keras

# Paths variables

DATA_DIR_ROOT = Path(
    r"C:\Users\thiba\OneDrive - CentraleSupelec\Mission_JCS_IA_peinture\files"
)
# DATA_DIR_ROOT = Path(r"/home/ec2-user/data")
IMAGES_DIR_PATH = DATA_DIR_ROOT / "images"
PATCHES_DIR_PATH = DATA_DIR_ROOT / "patches"
MASKS_DIR_PATH = DATA_DIR_ROOT / "labels_masks/all"
# MASKS_DIR = DATA_DIR_ROOT / "labels_masks"
OUTPUT_DIR_PATH = DATA_DIR_ROOT / "predictions"
CHECKPOINT_ROOT_DIR_PATH = DATA_DIR_ROOT / "checkpoints"
CHECKPOINT_DIR_PATH = DATA_DIR_ROOT / "checkpoints/2021_08_19__16_15_07"
IMAGE_PATH = DATA_DIR_ROOT / "images/_DSC0246/_DSC0246.jpg"
IMAGE_PATCH_PATH = DATA_DIR_ROOT / "patches/1/1/image/patch_1.jpg"
MASK_PATH = (
    DATA_DIR_ROOT
    / "labels_masks/all/1/feuilles-vertes/mask_1_feuilles-vertes__090f44ab03ee43d7aaabe92aa58b06c1.png"
)
OUTPUT_PATH = DATA_DIR_ROOT / "test.png"
# PREDICTIONS_PATH = DATA_DIR_ROOT / "predictions/_DSC0246/predictions_only/_DSC0246_predictions__model_2021_08_19__16_15_07__overlap_40.png"
PREDICTIONS_PATH = DATA_DIR_ROOT / "predictions/test.png"

# Palette & mapping related variables


def turn_hexadecimal_color_into_nomalized_rgb_list(hexadecimal_color: str) -> [int]:
    hexadecimal_color = hexadecimal_color.lstrip("#")
    return tuple(int(hexadecimal_color[i:i+2], 16) / 255 for i in (0, 2, 4))


def turn_hexadecimal_color_into_rgb_list(hexadecimal_color: str) -> [int]:
    hexadecimal_color = hexadecimal_color.lstrip("#")
    return tuple(int(hexadecimal_color[i:i+2], 16) for i in (0, 2, 4))


MAPPING_CLASS_NUMBER = {
    "background": 0,
    "poils-cheveux": 1,
    "vetements": 2,
    "peau": 3,
    "bois-tronc": 4,
    "ciel": 5,
    "feuilles-vertes": 6,
    "herbe": 7,
    "eau": 8,
    "roche": 9
}  # Maps each labelling class to a number
PALETTE_HEXA = {
    0: "#DCDCDC",  #gainsboro
    1: "#8B6914",  #goldenrod4
    2: "#BF3EFF", #darkorchid1
    3: "#FF7D40",  #flesh
    4: "#E3CF57",  #banana
    5: "#6495ED",  #cornerflowblue
    6: "#458B00",  #chartreuse4
    7: "#7FFF00",  #chartreuse1
    8: "#00FFFF",  #aqua
    9: "#FF0000"  #red
}
PALETTE_RGB_NORMALIZED = {
    key: turn_hexadecimal_color_into_nomalized_rgb_list(value) for key, value in PALETTE_HEXA.items()
}
PALETTE_RGB = {
    key: turn_hexadecimal_color_into_rgb_list(value) for key, value in PALETTE_HEXA.items()
}

# LabelBox related variables

MASK_URL = "https://api.labelbox.com/masks/feature/ckph5r33g00043a6dklihalmq?token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VySWQiOiJja3BneTBhZDc4OXAwMHk5dzZlcWM2bzNlIiwib3JnYW5pemF0aW9uSWQiOiJja3BneTBhY3U4OW96MHk5dzNrcW43MGxmIiwiaWF0IjoxNjIyNzQwNjczLCJleHAiOjE2MjUzMzI2NzN9.VeR0ot2_MAkY769kcXSz8RWqRguopgO1rlbRIGwZWV0"
JSON_PATH = Path(
    "C:/Users/thiba/OneDrive - CentraleSupelec/Mission_JCS_IA_peinture/labelbox_export_json/export-2021-07-26T14_40_28.059Z.json"
)
# Values in a binary LabelBox mask
MASK_TRUE_VALUE = 255
MASK_FALSE_VALUE = 0

# Model parameters & hyperparameters

PATCH_SIZE = 256
INPUT_SHAPE = 256
BATCH_SIZE = 32 # 32 is a frequently used value
N_CLASSES = 9
N_EPOCHS = 10
N_PATCHES_LIMIT = 100
TEST_PROPORTION = 0.2
PATCH_OVERLAP = 40  # 20 not enough, 40 great
PATCH_COVERAGE_PERCENT_LIMIT = 75
ENCODER_KERNEL_SIZE = 3
LINEARIZER_KERNEL_SIZE = 3
N_CPUS = 4
# N_CPUS = 16
TARGET_HEIGHT = 2176
TARGET_WIDTH = 3264
PADDING_TYPE = "same"
# OPTIMIZER = "rmsprop"
OPTIMIZER = Adam(lr=1e-4)
# LOSS_FUNCTION = "categorical_crossentropy"
LOSS_FUNCTION = keras.losses.categorical_crossentropy
# METRICS = ["accuracy", keras.metrics.MeanIoU]
# METRICS = [keras.metrics.MeanIoU(N_CLASSES)]
METRICS = [keras.metrics.categorical_accuracy, keras.metrics.MeanIoU(N_CLASSES)]
DOWNSCALE_FACTORS = (2, 2, 1)
