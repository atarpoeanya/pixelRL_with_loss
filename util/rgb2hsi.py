import numpy as np
import cv2
from math import pi

def toHSI(imageInput: np.ndarray):
    t = np.copy(imageInput)
    bgr = np.int32(cv2.split(t))
    
    blue = bgr[0]
    green = bgr[1]
    red = bgr[2]

    intensity = np.divide(blue + green + red, 3)

    minimum = np.minimum(np.minimum(red, green), blue)
    saturation = 1 - 3 * np.divide(minimum, red + green + blue)

    sqrt_calc = np.sqrt(((red - green) * (red - green)) + ((red - blue) * (green - blue))) + 0.00001
    

    sieve = [green < blue]

    hue = np.arccos((0.5 * ((red-green) + (red - blue)) / sqrt_calc))
    # print(hue, '\n')
    hue[np.all(np.array(sieve), axis=0)] = 2*pi - hue[np.all(np.array(sieve), axis=0)]
    # print(hue)
    # print('\n')

    hue = hue*180/pi

    # print("hue with rad")
    # print(hue)    
    # print('\n')

    hsi = cv2.merge((hue, saturation, intensity))
    return hsi


def backBGR(hsi):
  bgr = np.zeros(hsi.shape, np.int8)
  b,g,r = cv2.split(bgr)

  p = np.copy(hsi)
  h,s,i = cv2.split(p)
  h_no_rad = h * pi/180
  
 

#   h_no_rad = np.array([
#     [4.16789012, 2.34567890],
#     [0.98765432, 4.56789012]
#     # [3 ,2 ] [ 1, 3]
# ])

  # print(h_no_rad)
  # print('\n')
  hb = np.where((h_no_rad >= 4 * pi/3) & (h_no_rad < 2 * pi), 
                h_no_rad.copy() - 4 * pi/3 , 
                h_no_rad)
  # print(hb)
  # print('\n')
  hb = np.where((h_no_rad >= 2 * pi/3) & (h_no_rad < 4 * pi/3), h_no_rad.copy() - 2 * pi/3 , hb)
  # print(hb)
  # print('\n')
  
  # pool of pixel
  x = i * (1 - s)
  y = i *(1 + (s * np.cos(hb)) / (np.cos(pi/3 - hb)))
  z = 3*i - (x + y)

  sieveA = [(h_no_rad >= 4 * pi/3) & (h_no_rad < 2 * pi)]
  sieveB = [((h_no_rad >= 2 * pi/3) & (h_no_rad < 4 * pi/3))]

  b = x.copy()
  b[np.all(np.array(sieveA), axis=0)] = y[np.all(np.array(sieveA), axis=0)].copy()
  b[np.all(np.array(sieveB), axis=0)] = z[np.all(np.array(sieveB), axis=0)].copy()

  g = z.copy()
  g[np.all(np.array(sieveA), axis=0)] = x[np.all(np.array(sieveA), axis=0)].copy()
  g[np.all(np.array(sieveB), axis=0)] = y[np.all(np.array(sieveB), axis=0)].copy()

  r = y.copy()
  r[np.all(np.array(sieveA), axis=0)] = z[np.all(np.array(sieveA), axis=0)].copy()
  r[np.all(np.array(sieveB), axis=0)] = x[np.all(np.array(sieveB), axis=0)].copy()


  bgr = cv2.merge([b,g,r])
  bgr = np.uint8(np.round(bgr))
  
  return bgr

# imag = cv2.imread("./4_output.jpg")
# np.random.seed(1)
# img = np.random.randint(0, 256, size=(4, 4, 3), dtype=np.uint8)
# # img = cv2.resize(imag, (0, 0), None, .5, .5)
#   # toHSI(img)
# print(img)
# hsi = toHSI(img)
# bakc = backBGR(hsi)

# # print(img)
# # print('\n')
# # print(hsi)
# # print('\n')
# # print(bakc)


# # print(img)
# # print(hsi)
# numpy_horizontal_concat = np.concatenate((img, hsi), axis=1)
# cv2.imshow('image window', bakc)
# # add wait key. window waits until user presses a key
# cv2.waitKey(0)
# # and finally destroy/close all open windows
# cv2.destroyAllWindows()

