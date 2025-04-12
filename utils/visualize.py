from PIL import Image

def get_concat_h(im1, im2):
    # resize the image first wo the same height
    height = im1.height
    ratio = im1.height / im2.height
    im2_width = int(im2.width * ratio)
    im2 = im2.resize((im2_width, height))

    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst