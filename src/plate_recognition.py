import cv2
import numpy as np
import pytesseract
from PIL import Image


if __name__ == '__main__':
    img = cv2.imread('../res/test_img_6.jpg')
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bilateral_filtered_img = cv2.bilateralFilter(gray_img, 9, 75, 75)

    equal_histogram_img = cv2.equalizeHist(bilateral_filtered_img)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    morph_img = cv2.morphologyEx(equal_histogram_img, cv2.MORPH_OPEN, kernel, iterations=15)
    sub_morph_img = cv2.subtract(equal_histogram_img, morph_img)
    cv2.imshow("Subtracted", sub_morph_img)

    ret, binarized_img = cv2.threshold(sub_morph_img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    cv2.imshow("Binarized", binarized_img)

    canny = cv2.Canny(binarized_img, 250, 255)
    cv2.imshow("Canny", canny)

    canny = cv2.convertScaleAbs(canny)
    kernel = np.ones((3, 3), np.uint8)
    dilated_img = cv2.dilate(canny, kernel, iterations=1)
    cv2.imshow("Dilated", dilated_img)

    contours, hierarchy = cv2.findContours(dilated_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda cnt: cv2.contourArea(cnt), reverse=True)[:10]

    plate_contour = None
    for c in contours:
        perimeter = cv2.arcLength(c, True) 
        approx = cv2.approxPolyDP(c, 0.06 * perimeter, True)
        if len(approx) == 4:
            plate_contour = c
            break

    drawn_contours = cv2.drawContours(img, [plate_contour], -1, (0, 255, 0), 3)
    cv2.imshow("Contours", drawn_contours)

    mask = np.zeros(gray_img.shape, np.uint8)
    cv2.drawContours(mask, [plate_contour], 0, 255, -1)
    masked_contours = cv2.bitwise_and(img, img, mask=mask)
    cv2.imshow("Masked contours", masked_contours)
    cv2.imwrite('extracted_img.png',masked_contours)

    image_plate = cv2.imread('extracted_img.png')
    text = pytesseract.image_to_string(image_plate)
    print(text)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
