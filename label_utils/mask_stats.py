import numpy as np
from loguru import logger
from label_utils.mask_loader import load_mask

MASK_PATH = "C:/Users/thiba/OneDrive - CentraleSupelec/Mission_JCS_IA_peinture/masks/test/mask_angular_logo_lettre.png"
IMAGE_PATH = "C:/Users/thiba/OneDrive - CentraleSupelec/Mission_JCS_IA_peinture/images/angular_logo.png"


# TODO : changer les input des fonctions ci-dessus de mask_path à mask_array

# TODO : gather all the tests together at one place

def count_mask_value_occurences(mask_path):
    mask_array = load_mask_bis(mask_path)
    unique, counts = np.unique(
        mask_array[:, :, 0].flatten(), return_counts=True
    )
    # count_dict = dict(zip(unique, counts))
    percent_dict = dict(zip(unique, np.round(counts / counts.sum(), decimals=3)))
    logger.info(
        f"\nBackground percent : {percent_dict[0] * 100}"
        f"\nValue percent : {percent_dict[255] * 100}"
    )
    print({key: value for key, value in percent_dict.items() if value != 0})
    return percent_dict
