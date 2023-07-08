import os
import cv2 as cv
import numpy as np
import openpyxl

# Ruta donde se guardarán las imágenes
ruta_guardado = r'C:\Users\crist\Desktop\Proyecto_Reconocimiento\Ruidos\imagenes'

# Crear la carpeta si no existe
if not os.path.exists(ruta_guardado):
    os.makedirs(ruta_guardado)

# Pedir el nombre de la persona
nombre_persona = input("Ingrese el nombre de la persona: ")

# Ruta de la carpeta de la persona
ruta_persona = os.path.join(ruta_guardado, nombre_persona)

# Crear la carpeta de la persona si no existe
if not os.path.exists(ruta_persona):
    os.makedirs(ruta_persona)


#####################################################COMIENZO DEL CLASIFICADOR#####################################################################################################

# Cargar el clasificador de detección de rostros
ruidos = cv.CascadeClassifier(r'C:\Users\crist\Desktop\Proyecto_Reconocimiento\Ruidos\opencv-4.x\data\haarcascades\haarcascade_frontalface_default.xml')

# Inicializar la cámara
camara = cv.VideoCapture(0)
capturas = []  # Lista para almacenar las capturas de rostros
contador_capturas = 0  # Contador para el número de capturas realizadas

while True:
    _, captura = camara.read()  # Lo que se captura
    grises = cv.cvtColor(captura, cv.COLOR_BGR2GRAY)  # Escala de grises
    minSize = (20,20)
    caras = ruidos.detectMultiScale(grises, scaleFactor=1.1, minNeighbors=13, minSize=minSize)

    


    for (x, y, w, h) in caras:
        cv.rectangle(captura, (x, y), (x + w, y + h), (255, 0, 0), 2)
        rostro_capturado = grises[y:y+h, x:x+w]
        rostro_capturado = cv.resize(rostro_capturado, (160, 160), interpolation=cv.INTER_CUBIC)

    cv.imshow("Resultado rostro", captura)

    if cv.waitKey(1) == ord('s'):
        if contador_capturas < 25:
            # Tomar una captura cuando el usuario presione 's'
            captura_copia = rostro_capturado.copy()
            capturas.append(captura_copia)
            contador_capturas += 1
            print(f"Captura {contador_capturas} tomada")

    if contador_capturas == 25:
        break

camara.release()
cv.destroyAllWindows()

# Guardar las capturas en la carpeta de la persona
for i, captura in enumerate(capturas):
    nombre_archivo = f"{nombre_persona}_{i + 1}.jpg"
    ruta_archivo = os.path.join(ruta_persona, nombre_archivo)
    cv.imwrite(ruta_archivo, captura)

print(f"Imágenes guardadas en la carpeta: {ruta_persona}")

dataRUTA = r'C:\Users\crist\Desktop\Proyecto_Reconocimiento\Ruidos\imagenes'
listadata = os.listdir(dataRUTA)
ids = []
rostros_data = []

for id, folder in enumerate(listadata):
    ruta_completa = os.path.join(dataRUTA, folder)
    
    for archivo in os.listdir(ruta_completa):
        ruta_archivo = os.path.join(ruta_completa, archivo)
        imagen = cv.imread(ruta_archivo, 0) # Leer la imagen en escala de grises
        rostros_data.append(imagen)
        ids.append(id)

face_recognizer = cv.face.LBPHFaceRecognizer_create()


face_recognizer.train(rostros_data, np.array(ids))


face_recognizer.save('ENTRENAMIENTO_LBPH.xml')


face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('ENTRENAMIENTO_LBPH.xml')

ruidos = cv.CascadeClassifier(r'C:\Users\crist\Desktop\Proyecto_Reconocimiento\Ruidos\opencv-4.x\data\haarcascades\haarcascade_frontalface_default.xml')


camara = cv.VideoCapture(0)

workbook = openpyxl.Workbook()
sheet = workbook.active
sheet.cell(row=1, column=1).value = "Resultado"

while True:
    _, captura = camara.read()
    grises = cv.cvtColor(captura, cv.COLOR_BGR2GRAY)
    minSize = (20,20)
    caras = ruidos.detectMultiScale(grises, scaleFactor=1.1, minNeighbors=13, minSize=minSize)

    for (x, y, w, h) in caras:
        rostro_capturado = grises[y:y+h, x:x+w]
        rostro_capturado = cv.resize(rostro_capturado, (160, 160), interpolation=cv.INTER_CUBIC)
        resultado = face_recognizer.predict(rostro_capturado)

        if resultado[1] < 85:
            
            cv.putText(captura, nombre_persona, (x, y-20), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            #cv.rectangle(captura, (x, y), (x+w, y+h), (255, 0, 0), 2)
            nombre_persona = listadata[resultado[0]]
        else:
            nombre_persona = "Desconocido"
            cv.putText(captura, nombre_persona, (x, y-20), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv.rectangle(captura, (x, y), (x+w, y+h), (255, 0, 0), 2)
            sheet.append([nombre_persona])

    cv.imshow("Resultados", captura)

    if cv.waitKey(1) == ord('s'):
        break

camara.release()
cv.destroyAllWindows()
workbook.save("resultados.xlsx")