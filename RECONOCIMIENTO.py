import cv2 as cv
import os
import winsound
import time
import openpyxl

dataRUTA = r'C:\Users\crist\Desktop\Proyecto_Reconocimiento\Ruidos\imagenes'
listadata = os.listdir(dataRUTA)


face_recognizer = cv.face.LBPHFaceRecognizer.create()
face_recognizer.read('ENTRENAMIENTO_LBPH.xml')



ruidos = cv.CascadeClassifier(r'C:\Users\crist\Desktop\Proyecto_Reconocimiento\Ruidos\opencv-4.x\data\haarcascades\haarcascade_frontalface_default.xml')
camara=cv.VideoCapture(0)

# Crear un nuevo archivo Excel
workbook = openpyxl.Workbook()
sheet = workbook.active
sheet.cell(row=1, column=1).value = "Resultado"


while True:
    
    _,captura=camara.read()
    grises=cv.cvtColor(captura, cv.COLOR_BGR2GRAY)
    idcaptura=grises.copy()
    
    caras = ruidos.detectMultiScale(grises, 1.3, 5)
    
    
    for (x, y, e1, e2) in caras:
        
        rostrocapturado=idcaptura[y:y+e2,x:x+e1]
        rostrocapturado=cv.resize(rostrocapturado,(160,160),interpolation=cv.INTER_CUBIC)
        resultado=face_recognizer.predict(rostrocapturado)
        cv.putText(captura,'{}'.format(resultado),(x,y-5),1,1.3,(255,0,0),1,cv.LINE_AA)
      
        if resultado[1]<90:
            cv.putText(captura,'{}'.format(listadata[resultado[0]]),(x,y-20),1,1.3,(255,0,0),1,cv.LINE_AA)
            cv.rectangle(captura, (x, y), (x + e1, y + e2), (255, 0, 0), 2)
            nombre_persona = listadata[resultado[0]]
        
            #winsound.Beep(1000, 2000)
            #time.sleep(2) 
        else:    
            cv.putText(captura,"NO ENCOTRADO",(x,y-20),1,1.3,(255,0,0),1,cv.LINE_AA)
            cv.rectangle(captura, (x, y), (x + e1, y + e2), (255, 0, 0), 2)
            nombre_persona = "Desconocido"
        
        sheet.append([nombre_persona]) 
        
        
       
    cv.imshow("Resultados", captura)
    
    if cv.waitKey(1)==ord('s'):
        break
camara.release()
cv.destroyAllWindows()


workbook.save("resultados.xlsx")