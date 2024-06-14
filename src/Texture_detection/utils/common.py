# library 
from PIL import Image 
import matplotlib.pyplot as plt 
from skimage.feature import graycomatrix,graycoprops
from skimage import io,color
# import cv2
import numpy as np 
import pandas as pd 
import datetime
from patchify import patchify
from itertools import permutations
from itertools import combinations_with_replacement
from sklearn.cluster import KMeans
# from PIL import Image

# # inputs
# img_path=""
# PATCH_SIZE=21
# no_of_clusters= 4

# Reading the image

def read(file):
    img= io.imread(file)
    return img

# convert into gray scale
def convert_to_gray(img):
    gray= color.rgb2gray(img)
    return gray 

# making patches out of it
def ExtractFeature(img, patch_size,stride,GLCM_bin_size):
    diss_sim = []
    corr = []
    homogen = []
    energy = []
    contrast = []
    uniformity=[]
    
# creating patches
    patches= patchify(img,patch_size,stride)
    patches= np.int8(patches*(GLCM_bin_size-1))
    no_of_patches=len(patches)
    for i in range(no_of_patches):
        for j in range(no_of_patches):
            glcm = graycomatrix(patches[i][j], distances=[1], angles=[0], levels=GLCM_bin_size,
                                    symmetric=True, normed=True)
            diss_sim.append(graycoprops(glcm, 'dissimilarity',)[0, 0]) #[0,0] to convert array to value
            corr.append(graycoprops(glcm, 'correlation')[0, 0])
            homogen.append(graycoprops(glcm, 'homogeneity')[0, 0])
            energy.append(graycoprops(glcm, 'energy')[0, 0])
            contrast.append(graycoprops(glcm, 'contrast')[0, 0])
            uniformity.append(graycoprops(glcm,'ASM')[0,0]) 
            

    return diss_sim,corr,homogen,energy,contrast,uniformity

def CentreCoord(img,patch_size):
    size_of_img=img.shape[0]
    first_pt=patch_size//2+1
    last_pt=(size_of_img)-(patch_size//2)+2   
    number=np.arange(first_pt,last_pt)
    
    centre_coords1= list(permutations(number,2))
    centre_coords2= list(combinations_with_replacement(number,2))
    centre_coords= sorted(set(centre_coords1+centre_coords2))
    
    return centre_coords

def Colouring(blank,df,patch_size,pixels,clusters): 
    cluster_labels=np.arange(0,clusters)
    colours=[(255,0,0),(0,255,0),(0,0,255),(0,255,255),(255,0,255),(255,255,0),(0,102,0),(102,0,0),(0,102,102),(204,153,255)]
    centre_itr=iter(pixels)
    
    for i in range(len(df)-1):
        for j in range(len(cluster_labels)):
            
            if df['labels'][i]==cluster_labels[j]:
                blank[next(centre_itr)]=colours[j]

    coloured_img=blank
    
    return coloured_img


def pipeline(img,patch_size,stride,GLCM_bin_size,n_clusters):
    #blank image
    blank=np.zeros([img.shape[0],img.shape[0],3],dtype='float')
    
    t1=datetime.datetime.now()
    
#     extracting patches
    data=ExtractFeature(img,patch_size,stride,GLCM_bin_size)
    
    
#     storing features in dataframe
    data={'diss_sim':data[0],'corr':data[1],'homogen':data[2],
         'energy':data[3],'contrast':data[4],
         'uniformity':data[5]}    
    df=pd.DataFrame(data)
    
#     kmeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=0) 
    kmeans.fit(df)
    labels= kmeans.labels_
    df['labels']=labels 
    
#     fetching the centre pixels of patches
    centre=CentreCoord(img,patch_size)
#     colouring blank image
    coloured_img=Colouring(blank,df,patch_size,centre,n_clusters)
    
    t2=datetime.datetime.now()
    print(f"Time taken for Patch_size:{patch_size},Stride:{stride},Bin_size:{255/GLCM_bin_size}---", t2-t1)
    
    return coloured_img

