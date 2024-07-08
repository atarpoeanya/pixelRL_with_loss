import numpy as np
import sys
import cv2
import smallFunc

class State():
    def __init__(self, size):
        self.image = np.zeros(size,dtype=np.float32)

    def reset(self, x):
        self.image = x

    def step(self, act):

        bgr_t = np.transpose(self.image, (0,2,3,1))
        temp1 = np.zeros(bgr_t.shape, bgr_t.dtype)
        temp2 = np.zeros(bgr_t.shape, bgr_t.dtype)
        temp3 = np.zeros(bgr_t.shape, bgr_t.dtype)
        temp4 = np.zeros(bgr_t.shape, bgr_t.dtype)
        temp13 = np.zeros(bgr_t.shape, bgr_t.dtype)
        temp14 = np.zeros(bgr_t.shape, bgr_t.dtype)
        temp15 = np.zeros(bgr_t.shape, bgr_t.dtype)
        b, c, h, w = self.image.shape
        for i in range(0,b):
            if np.sum(act[i]==1) > 0:
                temp1[i] = smallFunc.contrast(bgr_t[i]+0.5, 0.95)
            if np.sum(act[i]==2) > 0:
                temp2[i] = smallFunc.contrast(bgr_t[i]+0.5, 0.95)
            if np.sum(act[i]==3) > 0:
                temp = cv2.cvtColor(bgr_t[i], cv2.COLOR_BGR2HSV)
                temp[1] *= 0.95
                temp3[i] = cv2.cvtColor(temp, cv2.COLOR_HSV2BGR)
            if np.sum(act[i]==4) > 0:
                temp = cv2.cvtColor(bgr_t[i], cv2.COLOR_BGR2HSV)
                temp[1] *= 1.05
                temp4[i] = cv2.cvtColor(temp, cv2.COLOR_HSV2BGR)
            if np.sum(act[i]==13) > 0:
                temp13[i] = smallFunc.clahe_hsv(bgr_t[i])
            if np.sum(act[i]==14) > 0:
                temp14[i] = smallFunc.umf(bgr_t[i])
            if np.sum(act[i]==15) > 0:
                temp15[i] = smallFunc.stretching(bgr_t[i])
        bgr1 = np.transpose(temp1, (0,3,1,2))
        bgr2 = np.transpose(temp2, (0,3,1,2))
        bgr3 = np.transpose(temp3, (0,3,1,2))
        bgr4 = np.transpose(temp4, (0,3,1,2))

        bgr5 = np.copy(self.image)
        bgr5 = bgr5 - 0.5*0.05
        bgr6 = np.copy(self.image)
        bgr6 = bgr6 + 0.5*0.05
        bgr7 = np.copy(self.image)
        bgr7[:,1:,:,:] *= 0.95
        bgr8 = np.copy(self.image)
        bgr8[:,1:,:,:] *= 1.05
        bgr9 = np.copy(self.image)
        bgr9[:,:2,:,:] *= 0.95
        bgr10 = np.copy(self.image)
        bgr10[:,:2,:,:] *= 1.05
        bgr11 = np.copy(self.image)
        bgr11[:,::2,:,:] *= 0.95
        bgr12 = np.copy(self.image)
        bgr12[:,::2,:,:] *= 1.05

        bgr13 = np.transpose(temp13, (0,3,1,2))
        bgr14 = np.transpose(temp14, (0,3,1,2))
        bgr15 = np.transpose(temp15, (0,3,1,2))
           

        
        act_3channel = np.stack([act,act,act],axis=1)
        self.image = np.where(act_3channel==1, bgr1, self.image)
        self.image = np.where(act_3channel==2, bgr2, self.image)
        self.image = np.where(act_3channel==3, bgr3, self.image)
        self.image = np.where(act_3channel==4, bgr4, self.image)
        self.image = np.where(act_3channel==5, bgr5, self.image)
        self.image = np.where(act_3channel==6, bgr6, self.image)
        self.image = np.where(act_3channel==7, bgr7, self.image)
        self.image = np.where(act_3channel==8, bgr8, self.image)
        self.image = np.where(act_3channel==9, bgr9, self.image)
        self.image = np.where(act_3channel==10, bgr10, self.image)
        self.image = np.where(act_3channel==11, bgr11, self.image)
        self.image = np.where(act_3channel==12, bgr12, self.image)
        self.image = np.where(act_3channel==13, bgr13, self.image)
        self.image = np.where(act_3channel==14, bgr14, self.image)
        self.image = np.where(act_3channel==15, bgr15, self.image)


