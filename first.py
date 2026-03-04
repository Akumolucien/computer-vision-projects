import cv2
import numpy as np


# Load an image
image = cv2.imread("IMAGES/people.jpg")



# Resize image
Resized=cv2.resize(image,[600,600])

blur= cv2.blur(Resized,(25,75))

#put text
text=cv2.putText(Resized,'cute',(50,50),
     cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)            


blur= cv2.blur(Resized,(25,75))

gaussian=cv2.GaussianBlur(image,(7,7),0)

#canny edge detection
egde=cv2.Canny(image,10,50)

#bilateral filter
bilateral=cv2.bilateralFilter(image,15,75,75)

cropped=Resized[20:100,100:200]
#translation function
def translate(Resized,x,y):
    transMat=np.float32([[1,0,x],[0,1,y]])
    dimensions=(Resized.shape[1],Resized.shape[0])
    return cv2.warpAffine(Resized, transMat, dimensions)

translate=translate(image,-100,100)
#cv2.imshow('translated', translate)

#Flipping
flip=cv2.flip(Resized,-1)
cv2.imshow('Flipped',flip)


# Display it
cv2.imshow("My Image", Resized)
#cv2.imshow("Blured image", blur)
#cv2.imshow("Cropped image", cropped)
#cv2.imshow("gaussian", gaussian)
#cv2.imshow('blur+edge', egde)
#cv2.imshow("bilateral+blur", bilateral)
#cv2.imshow("text", text)
cv2.waitKey(0)
cv2.destroyAllWindows()
