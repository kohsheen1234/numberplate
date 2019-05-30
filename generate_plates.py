# Several things to consider to create "real" NP dataset
# Download ttf font you want to use
# Install PIL
# This code will only generate simple number plates
# We further perform post-processing in Blender to create skewed/
# tilted/scaled and motion-blurred number plates.

from PIL import ImageFont, ImageDraw, Image  
import numpy as np 
import cv2
import random

# ASCII A to Z are 65 to 90

hel = [75, 25, 15, 130, 120] 
beb = [110, 2, 60, 135, 150]


#use a truetype font 
#font = ImageFont.truetype("Helvetica-Bold.ttf", 120)  
font = ImageFont.truetype("BebasNeue Bold.ttf", 150)  


rtc = 67
bias = 10
for r in range(rtc+1):
    if r < 4:
        for k in range(1000):
            if r < 10:
                number_plate_1 = "KA 0" + str(r)
            else:
                number_plate_1 = "KA " + str(r)
            number_plate_2 = (chr(random.randint(65, 90))+chr(random.randint(65, 90))+" " + str(random.randint(1000, 9999)))
            img = np.zeros((256, 512, 3), np.uint8)
            pil_img = Image.fromarray(img)
            draw = ImageDraw.Draw(pil_img)

            draw.text((75, 25), number_plate_1, font=font)  
            draw.text((15, 130), number_plate_2, font=font)
            cv2_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            cv2_img = cv2.bitwise_not(cv2_img)

            #cv2.imshow("number_plate", cv2_img)
            cv2.imwrite(number_plate_1+" "+number_plate_2+".png", cv2_img)
            #cv2.waitKey(10)
    else:
        for k in range(100):
            if r < 10:
                number_plate_1 = "KA 0" + str(r)
            else:
                number_plate_1 = "KA " + str(r)
            number_plate_2 = (chr(random.randint(65, 90))+chr(random.randint(65, 90))+" " + str(random.randint(1000, 9999)))
            img = np.zeros((256, 512, 3), np.uint8)
            pil_img = Image.fromarray(img)
            draw = ImageDraw.Draw(pil_img)

            draw.text((75, 25), number_plate_1, font=font)  
            draw.text((15, 130), number_plate_2, font=font)
            cv2_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            cv2_img = cv2.bitwise_not(cv2_img)

            #cv2.imshow("number_plate", cv2_img)
            cv2.imwrite(number_plate_1+" "+number_plate_2+".png", cv2_img)
            #cv2.waitKey(10)



cv2.destroyAllWindows()

