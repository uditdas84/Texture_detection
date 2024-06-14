# library 
from PIL import Image 
import matplotlib.pyplot as plt 
from skimage.feature import graycomatrix,graycoprops
from skimage import io
import cv2
import pandas as pd
import numpy as np
from src.Texture_detection.utils import common
import logging
import streamlit as st


# def covert_to_gray(img):
#     gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#     return gray 

# # crating patches
# def patch_centre(gray):
#     PATCH_SIZE=21
#     PATCH_CENTRE=[]
#     for i in range(1,330):  
#         for j in range(1,330):
#             PATCH_CENTRE.append(((PATCH_SIZE//2+1)+(i-1),(PATCH_SIZE//2+1)+(j-1)))                

#     return PATCH_CENTRE

# def create_patches(gray):
#     PATCHES=[]




st.title("Streamlit App for Texture detection")

# add a button to upload the image
uploaded_file= st.file_uploader("Choose an image...",type=["jpg"])

if uploaded_file is not None:
    # convert the file to opencv image 
    # file_bytes= np.asarray(bytearray(uploaded_file.read()),dtype=np.uint8)
    # logging.info("Reading the image")
    img = common.read(uploaded_file)

    # display the original image 
    st.image(img, channels="BGR",use_column_width=True)

    # 'Grayscale' button cteating
    if st.button("Convert to Grayscale"):
        # logging.info("Converting into grayscale")
        img_gray= common.convert_to_gray(img)
        st.image(img_gray,use_column_width=True)

    gray=common.convert_to_gray(img)
    patch_size=st.number_input("Patch size",min_value=1,max_value=100)
    GLCM_stride=1
    n_cluster=st.slider("No of cluster",min_value=1,max_value=10)
    bin_size=st.number_input("GLCM bin size",min_value=1,max_value=255)
    if st.button("Apply Texxture detection"):

        result=common.pipeline(gray,patch_size,GLCM_stride,bin_size,n_cluster)

        st.image(result,use_column_width=True,clamp=True)
    