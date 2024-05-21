import numpy as np
import cv2
import rgb2hsi

claheFilter_2 = cv2.createCLAHE(clipLimit=0.3, tileGridSize=(8,8))

def clahe_lab(image: np.ndarray, clip=0.3, tileSize=(8,8)):
  claheFilter_2 = cv2.createCLAHE(clip, tileSize)
  temp = np.zeros(image.shape, image.dtype)
  temp = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
  
  temp[...,0] = claheFilter_2.apply(temp[...,0])
  temp = cv2.cvtColor(temp, cv2.COLOR_LAB2BGR)

  return temp


def clahe_hsv(image: np.ndarray, clip=0.3, tileSize=(8,8)):
  claheFilter_2 = cv2.createCLAHE(clip, tileSize)
  temp = np.zeros(image.shape, image.dtype)
  temp = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
  
  temp[...,0] = claheFilter_2.apply(temp[...,0])
  temp = cv2.cvtColor(temp, cv2.COLOR_HSV2BGR)

  return temp
def umf(image: np.ndarray, SIGMA=0.8):
    img = np.copy(image)
    hsi = rgb2hsi.toHSI(img)
    # h,s,i = cv2.split(hsi) 
    
    filterGaus = hsi.copy()
    filterGaus[...,2] = cv2.GaussianBlur(filterGaus[...,2], (9,9), SIGMA)

    # hsi -= SIGMA*filterGaus
    edge = hsi[...,2] - filterGaus[...,2]

    hsi[...,2] += SIGMA * edge
    # hsi = cv2.merge([h,s,i])
    return rgb2hsi.backBGR(hsi)

def stretching(image: np.ndarray, **kwargs):
    # Split the image into channels


    b, g, r = cv2.split(image)

    
    # Apply dark stretching to each channel
    b_stretched = dark_channel(b, **kwargs)
    g_stretched = dark_channel(g, **kwargs)
    r_stretched = dark_channel(r, **kwargs)
    
    # Merge the channels back into an RGB image
    stretched_image = cv2.merge([b_stretched, g_stretched, r_stretched])
    
    return stretched_image



def dark_channel(channel: np.ndarray, gamma=-20, multi=1):
    # Find minimum and maximum pixel values

    I = channel.copy()
    FS,th2 = cv2.threshold(I,0,255,cv2.THRESH_OTSU+cv2.THRESH_BINARY)

    lower = I[I < FS]
    upper = I[I > FS]
    
    max_upper, min_upper = max(upper, default=0), min(upper, default=0)
    max_lower, min_lower = max(lower, default=0), min(lower, default=0)

    # print(max_upper, min_upper, max_lower, min_lower)
    upper_f =lambda i: ((FS + gamma) * multi) + (i - min_upper) * ((255 - ((FS + gamma) * multi)) / ((max_upper - min_upper) + 0.00001))
    lower_f =lambda i: (i - min_lower) * (((FS + gamma) * multi) / ((max_lower - min_lower) + 0.00001))
    
    # If pixel value is more than threshold
    new_I = np.where(I > FS, 
                        upper_f(I),
                        lower_f(I)
                     )


    stretched_channel = np.uint8(new_I)

    # print(max_upper, min_upper, max_lower, min_lower)

    return stretched_channel


def lower_contrast(image, contrast_factor=0.3, b=0):
    """
    Lower the contrast of an image.
    """
    temp = np.zeros(image.shape, image.dtype)
    b += int(round(255*(1-contrast_factor)/2))
    temp = cv2.addWeighted(image, contrast_factor, image, 0, b)
    return temp

