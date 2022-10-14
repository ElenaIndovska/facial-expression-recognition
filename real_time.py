from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np
from model import FacialExpressionModel
from PIL import Image

face_classifier = cv2.CascadeClassifier(r'D:\fakultet\face-emotion-recognition-main\diplomska\haarcascade_frontalface_default.xml')
model = FacialExpressionModel("MobileNetV2_high_intensity_78.9%.h5", "MobileNetV2_high_intensity_78.9%.h5")
font = cv2.FONT_HERSHEY_SIMPLEX

emotion_labels = ["Eyebrow raise", "Frown", "Smile", "Squeezed eyes"]

cap = cv2.VideoCapture(0)



while True:
    _, frame = cap.read()
    labels = []
    fr = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = face_classifier.detectMultiScale(fr)

    for (x,y,w,h) in faces:
        fc = fr[y:y+h, x:x+w]
        img = cv2.resize(fc, (224, 224))
        img = img.reshape((224,224,3))
        pred = model.predict_expression(img[np.newaxis, :, :, :])

        cv2.putText(frame, pred, (x, y), font, 1, (255, 255, 0), 2)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

       
    cv2.imshow('Expression Detector',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()