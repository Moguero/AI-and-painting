## [FULL DESCRIPTION WRITING ONGOING]

### Summary

### Methodology

1. Creation of the dataset.

a. Download labels masks from LabelBox.

The PNG labels masks can be downloaded with the command : 

python labelbox_utils/mask_downloader.py --json-path "<path_to_json>" --output-dir-path "<path_to_output_dir>"

b. Merge the masks together

c. Reorganize the masks folder structure.

d. Upload images with masks (images_utils).

e. stack image masks (save_all_categorical_masks)
