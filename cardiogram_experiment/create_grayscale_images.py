"""
Create grayscale images from coloured ones.

Needs minor modifications since I moved directories and files around!
"""
import os
from PIL import Image
from misc import IMAGES_DIR, get_stimuli_directories

DIRECTORIES = get_stimuli_directories()
PATHS = [IMAGES_DIR + d for d in DIRECTORIES]

for p, path in enumerate(PATHS):
    for i, image in enumerate(os.listdir(path)):
        if os.path.splitext(os.path.join(path, image))[-1].lower() == '.jpg':
            img = Image.open(os.path.join(path, image)).convert('LA') \
                .convert('RGB')
            filename = image.split('.')[0]
            filename += '_grayscale.jpg'
            grayscale_path = path.split('/')
            if grayscale_path[-1] == 'Pretraining':
                grayscale_path[-1] += '_grayscale'
            else:
                grayscale_path[-2] += '_grayscale'
            grayscale_path.append(filename)
            grayscale_path = '/'.join(grayscale_path)
            new_directory = '/'.join(os.path.splitext(
                grayscale_path)[0].split('/')[:-1])
            try:
                os.makedirs(new_directory)
            except OSError:
                pass
            img.save(grayscale_path)
