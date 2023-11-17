import os
import numpy as np
from PIL import Image

def img_getter(mask_path):
    assert os.path.isfile(mask_path)
    img = np.array(Image.open(mask_path).covert('RGB'))
    return img

