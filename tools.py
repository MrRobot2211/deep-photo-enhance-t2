import tensorflow as tf
import numpy as np
import random, cv2, operator, os

print(tf.config.list_physical_devices('GPU'))

dir="/home/felipe/gans_enhancer/input/LPGAN/input/"
def rename_MIT_files(dir):

        for file in os.listdir(dir):

                os.rename(dir+file,dir+file.split("-")[0]+".tif")


        for file in os.listdir(dir):

                os.rename(dir+file,dir+file.split(".")[0]+".tif")

from PIL import Image
def resize_image(dir,max_length = 512) :
        for file in os.listdir(dir):
        
                image = Image.open(dir+file)
                print(image.size)
                max_size = np.argmax(image.size )
                width, height = image.size 
                factor=max_length/image.size[max_size]
                new_image = image.resize((int(factor*width), int(factor*height)), Image.ANTIALIAS)
                image.close()
                new_image.save(dir+file)
