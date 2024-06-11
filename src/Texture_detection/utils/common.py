# library 
import cv2


# Reading the image


# convert into gray scale
def covert_to_gray(img):
    gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    return gray 

# making patches out of it

