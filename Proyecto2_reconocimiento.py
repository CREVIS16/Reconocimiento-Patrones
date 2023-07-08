import cv2 as cv
import os
import numpy as np

dataRUTA = r'C:\Users\crist\Desktop\Proyecto_Reconocimiento\Ruidos\imagenes'
listadata = os.listdir(dataRUTA)
ids = []
rostros_data = []
id = 0

for fila in listadata:
    ruta_Completa = os.path.join(dataRUTA, fila)
    for archivo in os.listdir(ruta_Completa):
        ids.append(id)
        rostros_data.append(cv.imread(os.path.join(ruta_Completa, archivo), 0))
    id = id + 1

# Crear el reconocedor LBPH
face_recognizer = cv.face.LBPHFaceRecognizer.create()

# Entrenar el reconocedor
face_recognizer.train(rostros_data, np.array(ids))

# Guardar el modelo entrenado
face_recognizer.save('ENTRENAMIENTO_LBPH.xml')



#pip install opencv-contrib-python 
#pip install face_recognition  