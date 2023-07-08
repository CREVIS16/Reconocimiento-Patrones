import os
import cv2 as cv
import numpy as np

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
id=0
while True:
    _, captura = camara.read()  # Lo que se captura
    grises = cv.cvtColor(captura, cv.COLOR_BGR2GRAY)  # Escala de grises
    caras = ruidos.detectMultiScale(grises, 1.3, 5)  # porcentaje 5 == 50% y el 1 = 100%
    idcaptura=captura.copy()

    for (x, y, e1, e2) in caras:
        cv.rectangle(captura, (x, y), (x + e1, y + e2), (255, 0, 0), 2)
        rostrocapturado=idcaptura[y:y+e2,x:x+e1]
        rostrocapturado=cv.resize(rostrocapturado,(160,160),interpolation=cv.INTER_CUBIC)
        
        

    cv.imshow("Resultado rostro", captura)

    if cv.waitKey(1) == ord('s'):
        if contador_capturas < 25:
            # Tomar una captura cuando el usuario presione 's'
            captura_copia = rostrocapturado 
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
