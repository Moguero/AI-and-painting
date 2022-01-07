from tensorflow.keras.optimizers import Adam
from pathlib import Path
from tensorflow import keras
import tensorflow as tf

# Paths variables
from dataset_utils.image_utils import turn_hexadecimal_color_into_nomalized_rgb_list, \
    turn_hexadecimal_color_into_rgb_list, decode_image

local_machine = False

if local_machine:
    DATA_DIR_ROOT = Path(r"C:\Users\thiba\OneDrive - CentraleSupelec\Mission_JCS_IA_peinture\files")
    MASKS_DIR_PATH = DATA_DIR_ROOT / "labels_masks/all"
    REPORT_DIR_PATH = DATA_DIR_ROOT / r"reports/report_2022_01_06__17_43_17"
    IMAGES_DIR_PATH = Path(r"C:\Users\thiba\OneDrive - CentraleSupelec\Mission_JCS_IA_peinture\images\sorted_images\kept\all")
    TEST_IMAGES_DIR_PATH = DATA_DIR_ROOT / "test_images"
    DOWNSCALED_TEST_IMAGES_DIR_PATH = TEST_IMAGES_DIR_PATH / "downscaled_images" / "max"
    TEST_IMAGE_PATH = IMAGES_DIR_PATH / "_DSC0246.jpg"
    N_EPOCHS = 2
    N_PATCHES_LIMIT = 50
else:  # aws instance
    DATA_DIR_ROOT = Path(r"/home/data")
    MASKS_DIR_PATH = DATA_DIR_ROOT / "labels_masks"
    REPORT_DIR_PATH = DATA_DIR_ROOT / r"reports/report_2022_01_06__17_43_17"
    IMAGES_DIR_PATH = DATA_DIR_ROOT / "images"
    TEST_IMAGES_DIR_PATH = DATA_DIR_ROOT / "test_images"
    DOWNSCALED_TEST_IMAGES_DIR_PATH = TEST_IMAGES_DIR_PATH / "downscaled_images" / "max"
    TEST_IMAGE_PATH = IMAGES_DIR_PATH / "_DSC0246/_DSC0246.jpg"
    N_EPOCHS = 10
    N_PATCHES_LIMIT = 100

TEST_IMAGES_NAMES = [
    "3.jpg",
    "4.jpg",
    "DSC_0097.jpg",
    "IMG_3083.jpg",
    "IMG_4698_2.jpg",
    "IMG_4724_2.jpg",
    "IMG_4831.jpg",
    "IMG_4939.jpg",
    "P1000724.jpg",
    "_DSC0036.jpg",
    "_DSC0064.jpg",
    "_DSC0103.jpg",
    "_DSC0177.jpg",
    "_DSC0201.jpg",
    "_DSC0231.jpg",
    "_DSC0235.jpg",
    "_DSC0241.jpg",
    "_DSC0245.jpg",
    "_DSC0257.jpg",
    "_DSC0300.jpg",
]

TEST_IMAGES_PATHS_LIST = [TEST_IMAGES_DIR_PATH / image_name for image_name in TEST_IMAGES_NAMES]
DOWNSCALED_TEST_IMAGES_PATHS_LIST = [DOWNSCALED_TEST_IMAGES_DIR_PATH / ("downscaled_max_" + image_name) for image_name in TEST_IMAGES_NAMES]

PATCHES_DIR_PATH = DATA_DIR_ROOT / "patches/256x256"
PREDICTIONS_DIR_PATH = DATA_DIR_ROOT / "predictions"
REPORTS_ROOT_DIR_PATH = DATA_DIR_ROOT / "reports"
# OUTPUT_PATH = DATA_DIR_ROOT / "test.png"
IMAGE_PATCH_PATH = DATA_DIR_ROOT / "patches/256x256/1/1/image/1_patch_1.jpg"
IMAGE_PATH = DATA_DIR_ROOT / "images/_DSC0246/_DSC0246.jpg"
MASK_PATH = (
    DATA_DIR_ROOT
    / "labels_masks/all/1/feuilles-vertes/mask_1_feuilles-vertes__090f44ab03ee43d7aaabe92aa58b06c1.png"
)
# PREDICTIONS_PATH = PREDICTIONS_DIR_PATH / "_DSC0246/predictions_only/_DSC0246_predictions__model_2021_08_19__16_15_07__overlap_40.png"
# PREDICTIONS_PATH = PREDICTIONS_DIR_PATH / "test.png"

# Palette & mapping related variables


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
MASK_URL = "https://api.labelbox.com/masks/feature/ckph5r33g00043a6dklihalmq?token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VySWQiOiJja3BneTBhZDc4OXAwMHk5dzZlcWM2bzNlIiwib3JnYW5pemF0aW9uSWQiOiJja3BneTBhY3U4OW96MHk5dzNrcW43MGxmIiwiaWF0IjoxNjIyNzQwNjczLCJleHAiOjE2MjUzMzI2NzN9.VeR0ot2_MAkY769kcXSz8RWqRguopgO1rlbRIGwZWV0"

# LabelBox related variables

JSON_PATH = Path(
    "C:/Users/thiba/OneDrive - CentraleSupelec/Mission_JCS_IA_peinture/labelbox_export_json/export-2021-07-26T14_40_28.059Z.json"
)

# Values in a binary LabelBox mask
MASK_TRUE_VALUE = 255
MASK_FALSE_VALUE = 0

# Model parameters & hyperparameters

PATCH_SIZE = 256
# BATCH_SIZE = 32  # 32 is a frequently used value
BATCH_SIZE = 8
N_CLASSES = 9
VALIDATION_PROPORTION = 0.2
TEST_PROPORTION = 0.1
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
OPTIMIZER = Adam(lr=1e-4)  # maybe put tf.Variable instead of the float to shut the warnings
LOSS_FUNCTION = "categorical_crossentropy"
# LOSS_FUNCTION = keras.losses.categorical_crossentropy
# METRICS = ["accuracy", keras.metrics.MeanIoU]
# METRICS = [keras.metrics.MeanIoU(N_CLASSES)]
METRICS = [keras.metrics.categorical_accuracy, keras.metrics.MeanIoU(N_CLASSES)]
DOWNSCALE_FACTORS = (6, 6, 1)
DATA_AUGMENTATION = False

# Physical parameters (in mm)

PHYSICAL_PIXEL_SIZE = 5
# todo : hardcode canvas_height and width directly into the predictions maker for testing
CANVAS_WIDTH = 4000
CANVAS_HEIGHT = 3000

MAX_WIDTH_PIXELS = 800
MAX_HEIGHT_PIXELS = 700

MIN_WIDTH_PIXELS = 160
MIN_HEIGHT_PIXELS = 140
