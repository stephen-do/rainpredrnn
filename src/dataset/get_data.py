import os
from tqdm import tqdm
import numpy as np
from PIL import Image

def get_data(configs):
    files = os.listdir(configs.path_root)
    files = sorted(files)
    sample_data = []
    for file in tqdm(files):
        real_path = os.path.join(configs.path_root, file)
        image = Image.open(real_path)
        image = image.convert('L')
        image = image.resize((configs.img_width, configs.img_width))
        data = np.expand_dims(np.array(image), -1).astype(np.float32)
        sample_data.append(data)
    sample_data = np.stack(sample_data, 0)
    return sample_data
