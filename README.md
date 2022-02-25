## AI & Painting

### Goal

This project aims at creating from scratch an AI pipeline designed to feed a robot with images and their classification maps, in order to assist a painter in his painting process 🎨



### Configuration
Firstly, clone this GitHub repository on your machine.
To install the environment I used for this challenge, you will need a virtual environment manager, for example [Anaconda](https://docs.anaconda.com/anaconda/install/).
In your terminal, navigate to the cloned repository and execute the following command line to reproduce the environment used :

```
conda env create -f environment_lin.yml
```

If you're on Linux, use environment_lin.yml file. On Windows, use environment_win.yml.

Once you're done, don't forget to remove the environment by running :

```
conda env remove -n ai-and-painting
```

### Methodology

1. Creation of the dataset.

a. Download labels masks from LabelBox.

The PNG labels masks can be downloaded with the command :

python labelbox_utils/mask_downloader.py --json-path "<path_to_json>" --output-dir-path "<path_to_output_dir>"

b. Merge the masks together

c. Reorganize the masks folder structure.

d. Upload images with masks (images_utils).

e. stack image masks (save_all_categorical_masks)

### Improvements

- track patches used for training with more detail (save used patches number, save minimized images)
- track data augmentation more closely during training
- increase number of unet encoder/decoder blocks