import numpy as np
import sys
import cv2
from util import smallFunc

class State():
    def __init__(self, size):
        self.image = np.zeros(size,dtype=np.float32)

    
    def reset(self, x):
        self.image = x
        size = self.image.shape
        prev_state = np.zeros((size[0],64,size[2],size[3]),dtype=np.float32)
        self.tensor = np.concatenate((self.image, prev_state), axis=1)

    def set(self, x):
        temp = np.copy(x)
        temp[:,0,:,:] /= 100
        temp[:,1,:,:] /= 127
        temp[:,2,:,:] /= 127
        self.tensor[:,:self.image.shape[1],:,:] = temp

    def step(self, act, inner_state, **kwargs):

        bgr1 = np.copy(self.image)
        # Global Stretching
        bgr1 = smallFunc.stretching(bgr1, **kwargs)

        bgr_t = np.transpose(self.image, (0,2,3,1))
        temp2 = np.zeros(bgr_t.shape, bgr_t.dtype)
        temp3 = np.zeros(bgr_t.shape, bgr_t.dtype)

        b, _, _, _ = self.image.shape
        # Pixel ise operation [
        #    -> CLAHE
        # ]
        for i in range(0,b):
            if np.sum(act[i]==2) > 0:
                temp2 = smallFunc.clahe_hsv(bgr_t[i], **kwargs)
            if np.sum(act[i]==3) > 0:
                temp3 = smallFunc.clahe_hsv(bgr_t[i], **kwargs)
        bgr2 = np.transpose(temp2, (0,3,1,2))
        bgr3 = np.transpose(temp3, (0,3,1,2))

        # Applying UMF globally
        bgr4 = np.copy(self.image)
        bgr4 = smallFunc.umf(bgr4, **kwargs)

        gaussian = np.zeros(self.image.shape, self.image.dtype)
        gaussian2 = np.zeros(self.image.shape, self.image.dtype)
        bilateral = np.zeros(self.image.shape, self.image.dtype)
        bilateral2 = np.zeros(self.image.shape, self.image.dtype)
        median = np.zeros(self.image.shape, self.image.dtype)
        box = np.zeros(self.image.shape, self.image.dtype)

        b, _, _, _ = self.image.shape
        for i in range(0,b):
            if np.sum(act[i]==self.move_range) > 0:
                gaussian[i,0] = cv2.GaussianBlur(self.image[i,0], ksize=(5,5), sigmaX=0.5)
            if np.sum(act[i]==self.move_range+1) > 0:
                bilateral[i,0] = cv2.bilateralFilter(self.image[i,0], d=5, sigmaColor=0.1, sigmaSpace=5)
            if np.sum(act[i]==self.move_range+2) > 0:
                median[i,0] = cv2.medianBlur(self.image[i,0], ksize=5)
            if np.sum(act[i]==self.move_range+3) > 0:
                gaussian2[i,0] = cv2.GaussianBlur(self.image[i,0], ksize=(5,5), sigmaX=1.5)
            if np.sum(act[i]==self.move_range+4) > 0:
                bilateral2[i,0] = cv2.bilateralFilter(self.image[i,0], d=5, sigmaColor=1.0, sigmaSpace=5)
            if np.sum(act[i]==self.move_range+5) > 0:
                box[i,0] = cv2.boxFilter(self.image[i,0], ddepth=-1, ksize=(5,5))
        # bgr6 = np.copy(self.image)
        # bgr6 = bgr6 + 0.5*0.05

        act_3channel = np.stack([act,act,act],axis=1)
        
        # Apply action based on prediction
        self.image = np.where(act_3channel==1, bgr1, self.image) 
        self.image = np.where(act_3channel==2, bgr2, self.image)
        self.image = np.where(act_3channel==3, bgr3, self.image)
        self.image = np.where(act_3channel==4, bgr4, self.image)
        # Apply denoising
        self.image = np.where(act_3channel==5, gaussian, self.image)
        self.image = np.where(act_3channel==6, bilateral, self.image)
        self.image = np.where(act_3channel==7, median, self.image)
        self.image = np.where(act_3channel==8, gaussian2, self.image)
        self.image = np.where(act_3channel==9, bilateral2, self.image)
        self.image = np.where(act_3channel==10, box, self.image)
        # self.image = np.where(act_3channel==11, bgr11, self.image)
        # self.image = np.where(act_3channel==12, bgr12, self.image)

        self.tensor[:,:self.image.shape[1],:,:] = self.image
        self.tensor[:,-64:,:,:] = inner_state


