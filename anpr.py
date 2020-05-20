import cv2
import imutils
import numpy as np
import pytesseract
from PIL import Image
import glob, os
import random as rng
from imutils import contours

#rng.seed(12345)

print(os.getcwd())
    os.chdir("VISOS")
print(os.getcwd())


# Resize image to specified size
def resize_image_to_specific_size(image):
    width = 620
    height = 480
    
    image = cv2.resize(image, (width, height))
    
    return image


# Morphological operations START #

# Dilation
def dilate(image, x, y):
    kernel = np.ones((x, y),np.uint8)
    return cv2.dilate(image, kernel, iterations = 1)
    
# Erosion
def erode(image, x, y):
    kernel = np.ones((x, y),np.uint8)
    return cv2.erode(image, kernel, iterations = 1)

# Opening - erosion followed by dilation
def opening_op(image, x, y):
    kernel = np.ones((x, y),np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

# Closing - dilation followed by erosion
def closing_op(image, x, y):
    kernel = np.ones((x, y),np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

def top_hat(image):
    kernel = np.ones((5, 5),np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)

def black_hat(image):
    kernel = np.ones((31, 31),np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)

# Morphological operations END #

# Contour searches START #

def contour_search_v1(gray, image):
    detected = 0
    
    cnts = cv2.findContours(image.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:100]
    screenCnt = None

    # Loop over our contours
    for c in cnts:
        # Approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.018 * peri, True)

        # If our approximated contour has four points, then
        # we can assume that we have found our screen
        if len(approx) == 4:
            screenCnt = approx
            break

    if screenCnt is None:
        detected = 0
        print ("No contour detected")
        return None
    else:
        detected = 1

    if detected == 1:
        cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 3)
        
        # Masking the part other than the number plate
        mask = np.zeros(gray.shape,np.uint8)
        new_image = cv2.drawContours(mask, [screenCnt], 0, 255, -1,)
        new_image = cv2.bitwise_and(image, image, mask = mask)

        # Now crop
        (x, y) = np.where(mask == 255)
        (topx, topy) = (np.min(x), np.min(y))
        (bottomx, bottomy) = (np.max(x), np.max(y))
        Cropped = gray[topx:bottomx + 1, topy:bottomy + 1]
        
        return Cropped
    
def contour_search_v2(image):
    cnts = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cnts, _ = contours.sort_contours(cnts, method = "left-to-right")


    for c in cnts:
        area = cv2.contourArea(c)
        x, y, w, h = cv2.boundingRect(c)
        center_y = y + h / 2
        if area > 3000 and (w > h):
            ROI = image[y: y + h, x: x + w]
            #ROI += image[y:y+h, x:x+w]
            #data = pytesseract.image_to_string(ROI, lang='eng', config='--oem 1 --psm 13')
            #data = pytesseract.image_to_string(ROI, config='-l eng --oem 1 --psm 13 tessedit_write_images=true -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
            plate = ROI
    
    return plate

# Contour searches END #
    

# Get number plate text from preprocessed image
def get_text_from_image(image):
    text = pytesseract.image_to_string(image, config='-l eng --oem 1 --psm 13 tessedit_write_images=true')
    
    return text
        
# Preproccess image with specified method values
def get_text_from_image_after_preproccessing(image, image_resizing, blur, gaussian_blur, median_blur, canny_x, canny_y, dilation, opening, closing, erosion, binarization, canny_after_mo, contour_search):
    
    
    # 1. Image resizing
    if image_resizing == 1:
        image = cv2.resize(image, (620, 480))
    elif image_resizing == 2:
        image = cv2.resize(image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
    else:
        image = imutils.resize(image, width=500)
    
    
    # 2. Photo conversion from colored to gray
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    
    # 3. If needed apply BLUR filter
    if blur != None:
        #print ("BLUR FILTER APPLIED")
        image = cv2.blur(image,(blur, blur))
    
    
    # 4. Apply GAUSSIAN BLUR filter
    image = cv2.GaussianBlur(image, (gaussian_blur, gaussian_blur), 0)
    
    
    # 5. Apply MEDIAN BLUR filter
    image = cv2.medianBlur(image, median_blur)
    
    
    
    ########## B. filter
    #image = cv2.bilateralFilter(image, 11, 23, 23)
    
    
    # 6. If needed apply CANNY EDGE detection (BEFORE morphological operations)
    if canny_after_mo == False:
        if canny_x != None and canny_y != None:
            #print ("CANNY EDGE DETECTION APPLIED BEFORE MO")
            image = cv2.Canny(image, canny_x, canny_y)
    
    
    # 7. If needed apply DILATION morphological operation
    image = dilate(image, dilation, dilation)
    
    
    # 8. If needed apply OPENING morphological operation
    if opening != None:
        #print ("OPENING MO APPLIED")
        image = opening_op(image, opening, opening)
    
    
    # 9. If needed apply CLOSING morphological operation
    if closing != None:
        #print ("CLOSING MO APPLIED")
        image = closing_op(image, closing, closing)
    
    
    # 10. If needed apply EROSION morphological operation
    if erosion != None:
        #print ("EROSION MO APPLIED")
        image = erode(image, erosion, erosion)
    
    
    # 11. If needed apply CANNY EDGE detection (AFTER morphological operations)
    if canny_after_mo == True:
        if canny_x != None and canny_y != None:
            #print ("CANNY EDGE DETECTION APPLIED AFTER MO")
            image = cv2.Canny(image, canny_x, canny_y)
    
    
    # 12. If needed apply binarization with THRESHOLD
    if binarization == "adaptiveThreshold":
        image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    elif binarization == "adaptiveThresholdGaussian":
        image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    
    # 13. Search for contours
    if contour_search == "v1":
        Cropped = contour_search_v1(gray, image)

        if Cropped is not None:
            text = get_text_from_image(Cropped)

            return text
    else:
        Countured = contour_search_v2(image)
        
        if Countured is not None:
            text = get_text_from_image(Countured)

            return text


# Main program
def main():
    for image in glob.glob("*.jpeg"):
        # Read original image from folder
        image = cv2.imread(image)
        
        # 1 program
        text = get_text_from_image_after_preproccessing(image,
                image_resizing = 1,
                blur = 7,
                gaussian_blur = 9,
                median_blur = 9,
                canny_x = 30,
                canny_y = 80,
                dilation = 3,
                opening = None,
                closing = None,
                erosion = None,
                binarization = None,
                canny_after_mo = False,
                contour_search = "v1"
        )
        
        print("1 program response:")
        print(text)
        
        # 2 program
        text = get_text_from_image_after_preproccessing(image,
                image_resizing = 1,
                blur = 7,
                gaussian_blur = 9,
                median_blur = 9,
                canny_x = 30,
                canny_y = 80,
                dilation = 3,
                opening = None,
                closing = None,
                erosion = None,
                binarization = None,
                canny_after_mo = False,
                contour_search = "v1"
        )
        
        print("2 program response:")
        print(text)

        # 3 program
        text = get_text_from_image_after_preproccessing(image,
                image_resizing = 2,
                blur = None,
                gaussian_blur = 9,
                median_blur = 9,
                canny_x = 30,
                canny_y = 80,
                dilation = 3,
                opening = 5,
                closing = None,
                erosion = 5,
                binarization = None,
                canny_after_mo = True,
                contour_search = "v1"
        )
        
        print("3 program response:")
        print(text)
        
        # 4 program
        text = get_text_from_image_after_preproccessing(image,
                image_resizing = 1,
                blur = 7,
                gaussian_blur = 9,
                median_blur = 9,
                canny_x = None,
                canny_y = None,
                dilation = 3,
                opening = None,
                closing = None,
                erosion = None,
                binarization = "adaptiveThreshold",
                canny_after_mo = False,
                contour_search = "v1"
        )
        
        print("4 program response:")
        print(text)
        
        # 5 program
        text = get_text_from_image_after_preproccessing(image,
                image_resizing = 1,
                blur = 7,
                gaussian_blur = 9,
                median_blur = 9,
                canny_x = 30,
                canny_y = 80,
                dilation = 5,
                opening = None,
                closing = None,
                erosion = None,
                binarization = None,
                canny_after_mo = False,
                contour_search = "v1"
        )
        
        print("5 program response:")
        print(text)
        
        # 6 program
        text = get_text_from_image_after_preproccessing(image,
                image_resizing = 3,
                blur = 7,
                gaussian_blur = 9,
                median_blur = 9,
                canny_x = None,
                canny_y = None,
                dilation = 3,
                opening = None,
                closing = None,
                erosion = None,
                binarization = "adaptiveThresholdGaussian",
                canny_after_mo = False,
                contour_search = "v2"
        )
        
        print("6 program response:")
        print(text)
    
    
main()