import numpy as np
import openslide
from openslide.deepzoom import DeepZoomGenerator
import os

np.random.seed(0)


wsi_path ='E:\\Panda\\WSI\\Panda\\'

def get_item(path):

    for filenames in os.listdir(path):
        print(filenames)
        result_path = 'E:\\Panda\\WSI\\output\\'
        result_path_img = os.path.join(result_path, filenames)
        if not os.path.isdir(result_path_img):
            os.makedirs(result_path_img)
        img_path =  os.path.join(wsi_path, filenames)
        slide = openslide.OpenSlide(img_path)
        data_gen = DeepZoomGenerator(slide, tile_size=256, overlap=0, limit_bounds=False)

        dim_num = data_gen.level_count - 1
        X_slide, Y_slide = data_gen.level_dimensions[dim_num]
        print(X_slide, Y_slide)
        target_size=256

        num_w = int(np.floor(X_slide/target_size))
        num_h = int(np.floor(Y_slide/target_size))
        print(num_w,num_h)

        for i in range(num_h):
            for j in range(num_w):
                img = data_gen.get_tile(dim_num ,(j,i))
                img.save(os.path.join(result_path_img, str(i)+'_'+str(j) + '.png'))
        slide.close()
get_item(wsi_path)