import cv2
import pytesseract

# Ruta al ejecutable de Tesseract (modifica la ruta según tu instalación)
pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

# Inicializar la cámara
cap = cv2.VideoCapture('http://192.168.1.128:8080/video')

# Cargar el clasificador Haar cascade para la detección de caras (puedes utilizar otro clasificador específico para la detección de credenciales si está disponible)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    # Capturar el fotograma de la cámara
    ret, frame = cap.read()

    # Convertir la imagen a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detectar caras en la imagen utilizando el clasificador Haar cascade
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Dibujar un cuadro delimitador alrededor de cada cara detectada
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Convertir la imagen a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Aplicar umbral adaptativo para mejorar la calidad del texto
    threshold = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 4)

    # Realizar OCR para reconocer el texto en la imagen
    text = pytesseract.image_to_string(threshold, lang='spa')

    # Dibujar rectángulos alrededor de las áreas con texto detectadas
    h, w = frame.shape[:2]
    boxes = pytesseract.image_to_boxes(threshold)
    for b in boxes.splitlines():
        b = b.split(' ')
        x, y, x2, y2 = int(b[1]), h - int(b[2]), int(b[3]), h - int(b[4])
        cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)

    # Mostrar la imagen en una ventana
    cv2.imshow('Captura de imagen', frame)

    # Salir si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        # Guardar la imagen capturada en un archivo
        cv2.imwrite('imagen.jpg', frame)
        break

# Liberar los recursos de la cámara
cap.release()
cv2.destroyAllWindows()

# Cargar la imagen de la credencial
image = cv2.imread('imagen.jpg')

# Convertir la imagen a escala de grises
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Aplicar umbral adaptativo para mejorar la calidad del texto
threshold = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 4)

# Realizar OCR para reconocer el texto en la imagen
text = pytesseract.image_to_string(threshold, lang='spa')

# Imprimir el texto extraído
print(text)
