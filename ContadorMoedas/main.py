import cv2
import numpy as np
from keras.models import load_model


video = cv2.VideoCapture(0, cv2.CAP_DSHOW)

model = load_model('keras_model.h5', compile=False)
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
class_names = ['1 Real', '25 Cent',  '50 Cent']

def preProcess(img):
    imgPre = cv2.GaussianBlur(img, (5, 5), 3)
    imgPre = cv2.Canny(imgPre, 90, 140)
    kernal = np.ones((4, 4), np.uint8)
    imgPre = cv2.dilate(imgPre, kernal, iterations=2)
    imgPre = cv2.erode(imgPre, kernal, iterations=2)
    return imgPre

def detectCoin (img):
    imgCoin = cv2.resize(img, (224, 224))
    imgCoin = np.asarray(imgCoin)
    imgCoinNormalize = (imgCoin.astype(np.float32) / 127.0) - 1
    data[0] = imgCoinNormalize
    prediction = model.predict(data)
    index = np.argmax(prediction)
    percent = prediction[0][index]
    class_name = class_names[index]
    return class_name, percent
    

while True:
    img = video.read()
    img = cv2.resize(img, (640, 480))
    imgPre = preProcess(img)
    countors, hi = cv2.findContours(imgPre, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    qtd  = 0
    
    for cnt in countors:
        area = cv2.contourArea(cnt)
        if area > 2000:
            x,y,w,h = cv2.boundingRect(cnt)
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            recorte = img[y:y +h, x:x +w]
            class trust : detectCoin(recorte)
            if trust > 0.75:
                cv2.putText (img, str(class_names), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0, 5, (0, 255, 0), 2)
                if class_names == '1 Real':
                    qtd += 1
                if class_names == '25 Cent':
                    qtd += 0.25
                if class_names == '50 Cent':
                    qtd += 0.50
                
        
    cv2.rectangle(img, (430,30), (600, 80), (0, 0, 255), -1)
    cv2.outText(img, f'R$ {qtd}', (440, 67), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    cv2.imshow('IMG', img)
    cv2.imshow('IMG Pre', imgPre)
    cv2.waitKey(1)