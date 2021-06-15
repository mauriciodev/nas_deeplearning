import numpy as np
from osgeo import ogr, gdal, gdal_array
import os,sys,math
import matplotlib.pyplot as plt
from PIL import Image
from sklearn import model_selection
#https://rockyshikoku.medium.com/train-deeplab-v3-with-your-own-dataset-13f2af958a75


"""def extract_patches(inputArray, patch_size, stride): #returns both the list of patches and an ordered set of patches.
    image = tf.expand_dims(inputArray,0) # To create the batch information
    patches = tf.image.extract_image_patches(image,[1,patch_size,patch_size,1],[1,stride,stride,1], rates=[1, 1, 1, 1],padding="VALID")
    patches=tf.reshape(patches, shape=[-1, patch_size, patch_size, image.shape[3]])
    res=[]
    for patch_id in range(patches.shape[0]):
        res.append(np.array(patches[1].eval(session=session)))
    tf.reset_default_graph()
    return res
"""
def extract_patches1(imgArray, patch_size, stride):
  # Read raster data as numeric array from file
  print(imgArray.shape)
  
  border_size=int((patch_size-stride)/2)
  #let's mirror the borders so we can actually run the entire image
  #imgArray2=np.pad(imgArray,((0,0),(border_size,border_size),(border_size,border_size)),mode='reflect')
  x_max,y_max,channels=imgArray.shape
  imagesArr=[]
  x=0
  while x_max>x+patch_size:
    y=0
    x_0=x
    x_1=x_0+patch_size
    while y_max>y+patch_size:
        y_0=y
        y_1=y_0+patch_size
        subArray=imgArray[x_0:x_1,y_0:y_1,:]
        imagesArr.append(subArray)
        y+=stride
    x+=stride
  return imagesArr

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

#if __name__=="__main__":


patch_size = 256
stride = 256-32 #using 32 pixels of overlap

imagesId=[3,4]
band_images_pattern="TLS_BDSD_NIRRG/TLS_BDSD_NIRRG_{:02d}.tif"
label_images_pattern="TLS_indMap/TLS_indMap_{:02d}.tif"

outputdir=os.getcwd()+"/TLS"

#for imageName in os.listdir():
imageList=[]


for imageid in imagesId:
    band_image=band_images_pattern.format(imageid)
    label_image=label_images_pattern.format(imageid)

    if not (os.path.exists(band_image) and os.path.exists(label_image)):
        print(f"Some images were not found: {band_image} {label_image}")
        continue
    baseFileName=os.path.split(band_image)[-1][:-4]

    bandArray = gdal_array.LoadFile(band_image)
    bandArray = np.moveaxis(bandArray, 0, -1)
    band_patches=extract_patches1(bandArray,patch_size,stride)
    labelsArray = gdal_array.LoadFile(label_image) #this is a grayscale image
    labelsArray = labelsArray.reshape(labelsArray.shape+(1,))
    #labelsArray = np.moveaxis(labelsArray, 0, -1)
    label_patches=extract_patches1(labelsArray,patch_size,stride)
    
    classRawDir=os.path.join(outputdir,'SegmentationClassRaw')
    jpegImagesDir=os.path.join(outputdir,'JPEGImages')
    imageSetDir=os.path.join(outputdir,'ImageSet')
    os.makedirs(classRawDir, exist_ok=True)
    os.makedirs(jpegImagesDir, exist_ok=True)
    os.makedirs(imageSetDir, exist_ok=True)
    print(len(band_patches))
    

    
    for patch_id in range(len(band_patches)):

        patch_array=np.array(band_patches[patch_id])
        imageId=f'{baseFileName}_{patch_id:03}'
        imageList.append(imageId)
        #outputImage = np.moveaxis(patch_array, -1, 0)
        im = Image.fromarray(patch_array)
        im.save(os.path.join(jpegImagesDir,imageId+'.png'), quality=100, subsampling=0)
        grayPatch=np.array(label_patches[patch_id]) #returning to 2D because it's grayscale
        im = Image.fromarray(np.squeeze(grayPatch) , 'L') 
        im.save(os.path.join(classRawDir,imageId+'.png'), quality=100, subsampling=0)
        #gdal_array.SaveArray(outputImage,os.path.join(outputdir,f'{baseFileName}_{patch_id}_.jpg'),  format="JPEG")
    
    with open(os.path.join(imageSetDir,'trainval.txt'),'w') as f:
        for image in imageList:
            f.write(image+'\n')
    #train.txt  trainval.txt  val.txt
    (train, val) = model_selection.train_test_split(imageList, test_size=0.20, random_state=42)
    with open(os.path.join(imageSetDir,'train.txt'),'w') as f:
        for image in train:
            f.write(image+'\n')
    with open(os.path.join(imageSetDir,'val.txt'),'w') as f:
        for image in val:
            f.write(image+'\n')
    
    #this is just an example of how to restore the images
    #outputImage=rebuildImage(band_patches,bandArray.shape,patch_size,stride)
    #outputImage = np.moveaxis(outputImage, -1, 0)
    #gdal_array.SaveArray(outputImage,os.path.join(os.getcwd(),'teste.tif'))


    #print(bandArray.shape)
    #print(outputImage.shape)
