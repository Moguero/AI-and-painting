# Maps each labelling class to a number
MAPPING_CLASS_NUMBER = {
    "background": 0,
    "poils_cheveux": 1,
    "vetements": 2,
    "peau": 3,
    "bois_tronc": 4,
    "ciel": 5,
    "feuilles_vertes": 6,
    "herbe": 7,
    "eau": 8,
    "roche": 9
}


PALETTE_HEXA = {
    "background": "#DCDCDC",  #gainsboro
    "poils_cheveux": "#8B6914",  #goldenrod4
    "vetements": "#BF3EFF", #darkorchid1
    "peau": "#FF7D40",  #flesh
    "bois_tronc": "#E3CF57",  #banana
    "ciel": "#6495ED",  #cornerflowblue
    "feuilles_vertes": "#458B00",  #chartreuse4
    "herbe": "#7FFF00",  #chartreuse1
    "eau": "#00FFFF",  #aqua
    "roche": "#FF6103"  #cadmiumorange
}


# Values in a binary LabelBox mask
MASK_TRUE_VALUE = 255
MASK_FALSE_VALUE = 255


def turn_hexadecimal_color_into_rgb_list(hexadecimal_color: str) -> [int]:
    hexadecimal_color = hexadecimal_color.lstrip("#")
    return tuple(int(hexadecimal_color[i:i+2], 16) / 255 for i in (0, 2, 4))


PALETTE_RGB = {
    key: turn_hexadecimal_color_into_rgb_list(value) for key, value in PALETTE_HEXA.items()
}
