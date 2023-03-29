from einops import rearrange, reduce, repeat

def patchify(image, height_of_patch, width_of_patch, channel):
    batch, height_of_image, width_of_image, channel = image.shape
    assert height_of_image % height_of_patch == 0 and width_of_image % width_of_patch == 0, 'Image dimensions must be divisible by patch dimensions'
    
    num_patches_height = height_of_image // height_of_patch
    num_patches_width  = width_of_image // width_of_patch
    
    return rearrange(image, 'b (h1 h) (w1 w) c-> (b h1 w1) h w c', h1= num_patches_height, w1= num_patches_width)