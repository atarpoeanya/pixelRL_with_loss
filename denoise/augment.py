import cv2
import numpy as np
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

def apply_gaussian_blur(image, kernel_size):
    """Apply Gaussian blur to the image."""
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

def apply_motion_blur(image, kernel_size):
    """Apply motion blur to the image."""
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[int((kernel_size - 1)/2), :] = np.ones(kernel_size)
    kernel = kernel / kernel_size
    return cv2.filter2D(image, -1, kernel)

def diminish_color_intensity(image, factor):
    """Diminish color intensity of the image."""
    temp = toHSI(image)
    temp[:, :, 2] = (temp[:, :, 2] * factor)
    return backBGR(temp)

def adjust_exposure(image, factor):
    """Adjust exposure of the image. Use factor > 1 to increase exposure, factor < 1 to decrease."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def process_image(image_path, blur_type='gaussian', blur_degree=5, color_intensity_factor=0.5, exposure_factor=1.2):
    """Process the image with given parameters."""
    image = cv2.imread(image_path)
    
    if blur_type == 'gaussian':
        image = apply_gaussian_blur(image, blur_degree)
    elif blur_type == 'motion':
        image = apply_motion_blur(image, blur_degree)

    image = diminish_color_intensity(image, color_intensity_factor)
    image = adjust_exposure(image, exposure_factor)

    return image

# Example usage
# input_image_path = './test_image\color_test.jpg'
input_image_path = './test_image/0_truth.png'
output_image_path = 'output.jpg'

processed_image = process_image(
    input_image_path,
    blur_type=  'gaussian',   #'motion',
    blur_degree=5,
    color_intensity_factor=1,
    exposure_factor= 1.9
)

# cv2.imwrite(output_image_path, processed_image)
cv2.imshow('Processed Image', processed_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

