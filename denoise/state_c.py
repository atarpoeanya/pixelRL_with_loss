import numpy as np
import sys
import cv2
import smallFunc

class State():
    def __init__(self, size, move_range):
        self.image = np.zeros(size,dtype=np.float32)
        self.move_range = move_range

    
    def reset(self, x):
        self.image = x
        # size = self.image.shape
        # prev_state = np.zeros((size[0],64,size[2],size[3]),dtype=np.float32)
        # self.tensor = np.concatenate((self.image, prev_state), axis=1)

    def step(self, act, gamma, multi,clip,SIGMA):

        bgr_t = np.transpose(self.image, (0,2,3,1))
        b, _, _, _ = self.image.shape

        temp1 = np.zeros(bgr_t.shape, bgr_t.dtype)
        temp2 = np.zeros(bgr_t.shape, bgr_t.dtype)
        temp3 = np.zeros(bgr_t.shape, bgr_t.dtype)
        temp4 = np.zeros(bgr_t.shape, bgr_t.dtype)

        # Global Stretching
        for i in range(0,b):
            temp1[i] = np.float32(smallFunc.stretching(np.uint8(bgr_t[i]), gamma=gamma, multi=multi))


        # Pixel ise operation [
        #    -> CLAHE
        # ]
        for i in range(0,b):
            # if np.sum(act[i]==1) > 0:
            if np.sum(act[i]==2) > 0:
                temp2[i] = np.float32(smallFunc.clahe_hsv(np.uint8(bgr_t[i]), clip))
            if np.sum(act[i]==3) > 0:
                temp3[i] = np.float32(smallFunc.clahe_hsv(np.uint8(bgr_t[i]), clip))
            
                
        bgr1 = np.transpose(temp1, (0,3,1,2))
        bgr2 = np.transpose(temp2, (0,3,1,2))
        bgr3 = np.transpose(temp3, (0,3,1,2))

        # Applying UMF globally
        for i in range(0,b):
            temp4[i] = np.float32(smallFunc.umf(np.uint8(bgr_t[i]), SIGMA))
        bgr4 = np.transpose(temp4, (0,3,1,2))
        
        # bgr4 = smallFunc.umf(bgr4, **kwargs)

        # gaussian = np.zeros(self.image.shape, self.image.dtype)
        # gaussian2 = np.zeros(self.image.shape, self.image.dtype)
        # bilateral = np.zeros(self.image.shape, self.image.dtype)
        # bilateral2 = np.zeros(self.image.shape, self.image.dtype)
        # median = np.zeros(self.image.shape, self.image.dtype)
        # box = np.zeros(self.image.shape, self.image.dtype)

        # b, _, _, _ = self.image.shape
        # for i in range(0,b):
        #     if np.sum(act[i]==self.move_range) > 0:
        #         gaussian[i,0] = cv2.GaussianBlur(self.image[i,0], ksize=(5,5), sigmaX=0.5)
        #     if np.sum(act[i]==self.move_range+1) > 0:
        #         bilateral[i,0] = cv2.bilateralFilter(self.image[i,0], d=5, sigmaColor=0.1, sigmaSpace=5)
        #     if np.sum(act[i]==self.move_range+2) > 0:
        #         median[i,0] = cv2.medianBlur(self.image[i,0], ksize=5)
        #     if np.sum(act[i]==self.move_range+3) > 0:
        #         gaussian2[i,0] = cv2.GaussianBlur(self.image[i,0], ksize=(5,5), sigmaX=1.5)
        #     if np.sum(act[i]==self.move_range+4) > 0:
        #         bilateral2[i,0] = cv2.bilateralFilter(self.image[i,0], d=5, sigmaColor=1.0, sigmaSpace=5)
        #     if np.sum(act[i]==self.move_range+5) > 0:
        #         box[i,0] = cv2.boxFilter(self.image[i,0], ddepth=-1, ksize=(5,5))
        # bgr6 = np.copy(self.image)
        # bgr6 = bgr6 + 0.5*0.05

        act_3channel = np.stack([act,act,act],axis=1)
        
        # Apply action based on prediction
        self.image = np.where(act_3channel==1, bgr1, self.image) 
        self.image = np.where(act_3channel==2, bgr2, self.image)
        self.image = np.where(act_3channel==3, bgr3, self.image)
        self.image = np.where(act_3channel==4, bgr4, self.image)

        
        # Apply denoising
        # self.image = np.where(act_3channel==5, gaussian, self.image)
        # self.image = np.where(act_3channel==6, bilateral, self.image)
        # self.image = np.where(act_3channel==7, median, self.image)
        # self.image = np.where(act_3channel==8, gaussian2, self.image)
        # self.image = np.where(act_3channel==9, bilateral2, self.image)
        # self.image = np.where(act_3channel==10, box, self.image)
        # self.image = np.where(act_3channel==11, bgr11, self.image)
        # self.image = np.where(act_3channel==12, bgr12, self.image)

        # self.tensor[:,:self.image.shape[1],:,:] = self.image
        # self.tensor[:,-64:,:,:] = inner_state


