from tensorflow.keras.optimizers import Adam
from pathlib import Path
from tensorflow import keras
import tensorflow as tf

# Paths variables
from dataset_utils.image_utils import turn_hexadecimal_color_into_nomalized_rgb_list, \
    turn_hexadecimal_color_into_rgb_list

local_machine = True

if local_machine:
    DATA_DIR_ROOT = Path(r"C:\Users\thiba\OneDrive - CentraleSupelec\Mission_JCS_IA_peinture\files")
    MASKS_DIR_PATH = DATA_DIR_ROOT / "labels_masks/all"
    CHECKPOINT_DIR_PATH = DATA_DIR_ROOT / "checkpoints/2021_10_10__18_16_44"
else:  # aws instance
    DATA_DIR_ROOT = Path(r"/home/data")
    MASKS_DIR_PATH = DATA_DIR_ROOT / "labels_masks"
    CHECKPOINT_DIR_PATH = DATA_DIR_ROOT / "checkpoints/2021_10_03__18_31_03"

IMAGES_DIR_PATH = DATA_DIR_ROOT / "images"
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
INPUT_SHAPE = 256
BATCH_SIZE = 32  # 32 is a frequently used value
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
OPTIMIZER = Adam(lr=1e-4)  # maybe put tf.Variable instead of the float to shut the warnings
# LOSS_FUNCTION = "categorical_crossentropy"
LOSS_FUNCTION = keras.losses.categorical_crossentropy
# METRICS = ["accuracy", keras.metrics.MeanIoU]
# METRICS = [keras.metrics.MeanIoU(N_CLASSES)]
METRICS = [keras.metrics.categorical_accuracy, keras.metrics.MeanIoU(N_CLASSES)]
DOWNSCALE_FACTORS = (6, 6, 1)
DATA_AUGMENTATION = False
