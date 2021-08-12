# Maps each labelling class to a number
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
}


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


# Values in a binary LabelBox mask
MASK_TRUE_VALUE = 255
MASK_FALSE_VALUE = 0


def turn_hexadecimal_color_into_nomalized_rgb_list(hexadecimal_color: str) -> [int]:
    hexadecimal_color = hexadecimal_color.lstrip("#")
    return tuple(int(hexadecimal_color[i:i+2], 16) / 255 for i in (0, 2, 4))


def turn_hexadecimal_color_into_rgb_list(hexadecimal_color: str) -> [int]:
    hexadecimal_color = hexadecimal_color.lstrip("#")
    return tuple(int(hexadecimal_color[i:i+2], 16) for i in (0, 2, 4))


PALETTE_RGB_NORMALIZED = {
    key: turn_hexadecimal_color_into_nomalized_rgb_list(value) for key, value in PALETTE_HEXA.items()
}

PALETTE_RGB = {
    key: turn_hexadecimal_color_into_rgb_list(value) for key, value in PALETTE_HEXA.items()
}
