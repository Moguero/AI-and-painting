import time
from loguru import logger
from dataset_utils.mask_utils import load_mask
import pandas as pd
from pathlib import Path

def transform_png_image_ndarray_to_dataframe(mask_path: Path):
    # transform png to numpy array to dataframe
    start_time = time.time()
    logger.info("Start the conversion from PNG to DataFrame...")
    img = load_mask(mask_path)
    multi_index = pd.MultiIndex.from_product([range(s) for s in img.shape], names=["x", "y", "channel"])
    img_df = pd.DataFrame({"img": img.flatten()}, index=multi_index).unstack("channel").rename(columns={0: "R", 1: "G", 2: "B", 3:"A"})

    # save dataframe for future use
    img_df.to_pickle()
    logger.info(f"\nConversion from PNG to DataFrame took : {round(time.time() - start_time, 3)} seconds to execute.")  # usual order of magnitude : 20s
    return img_df
