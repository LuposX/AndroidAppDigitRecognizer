from PIL import Image
from PIL import ImageEnhance
import os

# preprocess the image
def save_image(datadir):
    image = Image.open(datadir + "/photo.png").convert('L').point(lambda x: 0 if x<150 else 1)

    image = image.resize((128, 128))

    new_size = 28

    width, height = image.size

    left = (width - new_size)/2
    top = (height - new_size)/2
    right = (width + new_size)/2
    bottom = (height + new_size)/2
    image_crop = image.crop((left, top, right, bottom))

    # enhance the image
    # enhance = ImageEnhance.Sharpness(image_crop).enhance(2)
    # enhance_contrast = ImageEnhance.Contrast(enhance).enhance(2)

    # save the image
    image_crop.point(lambda x: x* 255).save(datadir + "/out.png")