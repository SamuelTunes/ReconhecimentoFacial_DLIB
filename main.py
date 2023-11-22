import cv2
import dlib
import face_recognition

def _css_to_rect(css):
    top, right, bottom, left = css
    return dlib.rectangle(left, top, right, bottom)

# Inicializar o detector de faces e o predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

images_person1 = ["dataset/rezende_001.png","dataset/rezende_002.png","dataset/rezende_003.png","dataset/rezende_004.png","dataset/rezende_005.png"]
images_person2 = ["dataset/joao_002.png","dataset/joao_003.png","dataset/joao_004.png","dataset/joao_005.png",]

#encodings_person2 = [face_recognition.face_encodings(face_recognition.load_image_file(img)) for img in images_person2]
#for idx, encodings in enumerate(encodings_person2):
#    if not encodings:
#       print(f"No face found in image {images_person2[idx]}")
#    else:
#      print(f"Face encoding for image {images_person2[idx]}: {encodings[0]}")

encodings_person1 = [face_recognition.face_encodings(face_recognition.load_image_file(img))[0] for img in images_person1]
encodings_person2 = [face_recognition.face_encodings(face_recognition.load_image_file(img))[0] for img in images_person2]

known_encodings = encodings_person1 + encodings_person2
known_labels = ["Rezende"] * len(encodings_person1) + ["Joao"] * len(encodings_person2)

video_capture = cv2.VideoCapture("video/video2F.mp4")

frame_skip = 10
frame_count = 0

while True:
    # Capturar frame por frame
    ret, frame = video_capture.read()

    # Se não houver mais frames, sair do loop
    if not ret:
        break
    
    frame_count += 1

    # Pular frames se necessário
    if frame_count % frame_skip != 0:
        continue
    
    # Converter para escala de cinza  
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Obter as localizações das faces no frame
    face_locations = face_recognition.face_locations(frame)
    for face_location in face_locations:
        # Obter as coordenadas da caixa delimitadora
        top, right, bottom, left = face_location

        # Obter a codificação do rosto na região atual
        face_encoding = face_recognition.face_encodings(frame, [face_location])

        # Verificar se é uma face conhecida
        if face_encoding:
            matches = face_recognition.compare_faces(known_encodings, face_encoding[0], tolerance=0.5)
            name = "Desconhecido"

            # Se houver correspondência, obter o rótulo correspondente
            if True in matches:
                first_match_index = matches.index(True)
                name = known_labels[first_match_index]

            # Desenhar um retângulo ao redor da face e exibir o rótulo
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Mostrar o frame resultante
    cv2.imshow("Video", frame)
    
    # Sair do loop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


video_capture.release()
cv2.destroyAllWindows()
