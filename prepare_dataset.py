import numpy as np
from osgeo import ogr, gdal, gdal_array
import tensorflow as tf
import os,sys,math
import matplotlib.pyplot as plt
from PIL import Image

#https://rockyshikoku.medium.com/train-deeplab-v3-with-your-own-dataset-13f2af958a75


"""Adds a border of (patch_size-stride)/2 and slices the input array."""
def extract_patches(inputArray, patch_size, stride): #returns both the list of patches and an ordered set of patches.
    border_size=int((patch_size-stride)/2)
    originalShape=inputArray.shape
    maxI=math.ceil(originalShape[1]/stride)
    maxJ=math.ceil(originalShape[0]/stride)
    extraPadJ=maxJ*stride-originalShape[0]
    extraPadI=maxI*stride-originalShape[1]
    image=np.pad(inputArray,((border_size,border_size+extraPadJ),(border_size,border_size+extraPadI),(0,0)),mode='symmetric')
    gdal_array.SaveArray(np.moveaxis(image, -1, 0),os.path.join(os.getcwd(),'teste_pad.tif'))
    #print(image.shape)
    image = tf.expand_dims(image,0) # To create the batch information
    patches = tf.image.extract_patches(image,[1,patch_size,patch_size,1],[1,stride,stride,1], rates=[1, 1, 1, 1],padding="VALID")
    return tf.reshape(patches, shape=[-1, patch_size, patch_size, image.shape[3]])

def extract_patches1(inputArray, patch_size, stride): #returns both the list of patches and an ordered set of patches.
    border_size=int((patch_size-stride)/2)
    originalShape=inputArray.shape
    maxI=math.ceil(originalShape[1]/stride)
    maxJ=math.ceil(originalShape[0]/stride)
    #print(image.shape)
    image = tf.expand_dims(inputArray,0) # To create the batch information
    patches = tf.image.extract_patches(image,[1,patch_size,patch_size,1],[1,stride,stride,1], rates=[1, 1, 1, 1],padding="VALID")
    return tf.reshape(patches, shape=[-1, patch_size, patch_size, image.shape[3]])

def rebuildImage(patches,originalShape,patch_size,stride):
    border_size=int((patch_size-stride)/2)
    cropped=tf.image.crop_to_bounding_box(patches,border_size,border_size,stride,stride)
    cols=int(np.ceil(originalShape[0]/stride))
    cropped_rows=np.array_split(cropped,cols)
    newRows=[]
    for row in cropped_rows:
        newRows.append(np.concatenate(row, 1))
    resImage=np.concatenate(newRows,0)
    resImage=tf.image.crop_to_bounding_box(resImage,0,0,originalShape[0],originalShape[1])
    return resImage

if __name__=="__main__":


    patch_size = 256
    stride = 256-32 #using 32 pixels of overlap

    imagesId=[3,4,7,8]
    band_images_pattern="TLS_BDSD_NIRRG/TLS_BDSD_NIRRG_{:02d}.tif"
    label_images_pattern="TLS_GT/TLS_GT_{:02d}.tif"
    
    outputdir=os.getcwd()+"/JPEG"
    
    #for imageName in os.listdir():
    for id in imagesId:
        band_image=band_images_pattern.format(id)
        label_image=label_images_pattern.format(id)

        if not (os.path.exists(band_image) and os.path.exists(label_image)):
            print(f"Some images were not found: {band_image} {label_image}")
            continue
        baseFileName=os.path.split(band_image)[-1][:-4]

        bandArray = gdal_array.LoadFile(band_image)
        bandArray = np.moveaxis(bandArray, 0, -1)
        band_patches=extract_patches1(bandArray,patch_size,stride)
        labelsArray = gdal_array.LoadFile(label_image)
        labelsArray = np.moveaxis(labelsArray, 0, -1)
        label_patches=extract_patches1(labelsArray,patch_size,stride)
        for patch_id, patch_array in enumerate(band_patches):
            #outputImage = np.moveaxis(patch_array, -1, 0)
            im = Image.fromarray(np.array(patch_array))
            im.save(os.path.join(outputdir,f'{baseFileName}_{patch_id}_.jpg'), quality=100, subsampling=0)
            im = Image.fromarray(np.array(label_patches[patch_id]))
            im.save(os.path.join(outputdir,f'{baseFileName}_{patch_id}_ref_.jpg'), quality=100, subsampling=0)
            #gdal_array.SaveArray(outputImage,os.path.join(outputdir,f'{baseFileName}_{patch_id}_.jpg'),  format="JPEG")

    
    #this is just an example of how to restore the images
    #outputImage=rebuildImage(band_patches,bandArray.shape,patch_size,stride)
    #outputImage = np.moveaxis(outputImage, -1, 0)
    #gdal_array.SaveArray(outputImage,os.path.join(os.getcwd(),'teste.tif'))


    #print(bandArray.shape)
    #print(outputImage.shape)
