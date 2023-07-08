import cv2 as cv
import numpy as np

ruidos = cv.CascadeClassifier(r'C:\Users\crist\Desktop\Proyecto_Reconocimiento\Ruidos\opencv-4.x\data\haarcascades\haarcascade_frontalface_default.xml')

camara=cv.VideoCapture(0)

while True:
    _,captura=camara.read() #Lo que se captura
    grises=cv.cvtColor(captura,cv.COLOR_BGR2GRAY) #Escala de grises
    cara=ruidos.detectMultiScale(grises,1.3,5)#porcentaje 5 == 50% y el 1 = 100%
    for(x ,y,e1,e2) in cara:
        cv.rectangle(captura,(x,y),(x+e1,y+e2),(255,0,0),2)
        
    cv.imshow("Resultado rostro",captura)

    if cv.waitKey(1)== ord('s'):
        break
 
    
camara.release()
cv.destroyAllWindows()   
   
    