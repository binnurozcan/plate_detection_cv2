import cv2
import numpy as np
import pytesseract
import imutils
import re

img = cv2.imread("img.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #siyah-beyaz
temp = cv2.bilateralFilter(gray, 1, 250, 250) #blurring
kenarlik = cv2.Canny(temp, 50, 200) #kenarları algılamak için

contours = cv2.findContours(kenarlik, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) #konturleri bul, kosegenler algılandı
cnts = imutils.grab_contours(contours)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10] #tersten sıralama yapılıyor, ilk 10 değer çekiliyor

screen = None

for c in cnts:
    epsilon = 0.018 * cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, epsilon, True)
    if len(approx) == 4: #dörtgen bir cisim
        screen = approx
        break

    if screen is None:
        detected = 0
        print("Şekil tespit edilemedi.")
    else:
        detected = 1
    if detected == 1:
        cv2.drawContours( img, [screen], -1, (0, 0, 255), 3)

mask = np.zeros(gray.shape, np.uint8) #grinin boyutunda shapeinde
result = cv2.drawContours(mask, [screen], 0, 255, -1)
result = cv2.bitwise_and(img, img, mask=mask) #and kestigi yerler
(x, y) = np.where(mask == 255) #x ve y koordinatları üzerinden maskelemeyi grey scale biçiminde kırpma
(topx, topy) = (np.min(x), np.min(y))
(botx, boty) = (np.min(x), np.min(y))
cropped = gray[topx:botx+1, topy:boty+1]

text = pytesseract.image_to_string(cropped, config='--psm 11')
text = re.sub('[^A-Za-z0-9^]', ' ', text)

cv2.imshpw("Original", img)
cv2.imshow("Cropped", cropped)
print("Plaka:" ,text)

cv2.waitKey(0)
cv2.destroyAllWindows()
