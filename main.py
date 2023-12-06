import cv2
import dlib
import face_recognition
import os
import tkinter as tk
from tkinter import filedialog, messagebox

window = tk.Tk()
window.title("Titulo Qualquer")
window.geometry("350x250")

video_path = tk.StringVar()
dataset_path = tk.StringVar()

def _css_to_rect(css):
    top, right, bottom, left = css
    return dlib.rectangle(left, top, right, bottom)

# Criando as funções para selecionar os arquivos
def select_video():
    path = filedialog.askopenfilename(title="Selecione o video", filetypes=[("Arquivos de video", "*.mp4 *.avi")])
    if path:
        video_path.set(path)
        video_entry.delete(0, tk.END)
        video_entry.insert(0, path)

def select_dataset():
    path = filedialog.askdirectory(title="Selecione o dataset")
    if path:
        dataset_path.set(path)
        dataset_entry.delete(0, tk.END)
        dataset_entry.insert(0, path)

# Criando a função para enviar os caminhos e executar o programa
def send_paths():
    try:
        video_capture = cv2.VideoCapture(video_path.get())

        # Listar todos os arquivos na pasta de dataset
        image_files = [f for f in os.listdir(dataset_path.get()) if f.lower().endswith((".png", ".jpg", ".jpeg"))]

        # Codificações e rótulos conhecidos
        known_encodings = []
        known_labels = []

        # Processar cada imagem na pasta
        for image_file in image_files:
            image_path = os.path.join(dataset_path.get(), image_file)

            # Carregar a imagem e converter para RGB (face_recognition espera imagens RGB)
            image = face_recognition.load_image_file(image_path)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Obter localizações das faces na imagem
            face_locations = face_recognition.face_locations(rgb_image)

            if face_locations:
                face_encoding = face_recognition.face_encodings(rgb_image, face_locations)[0]

#Verificar se na imagem há algum rosto facial
#encodings_person2 = [face_recognition.face_encodings(face_recognition.load_image_file(img)) for img in images_person2]
#for idx, encodings in enumerate(encodings_person2):
#    if not encodings:
#       print(f"No face found in image {images_person2[idx]}")
#    else:
#      print(f"Face encoding for image {images_person2[idx]}: {encodings[0]}")

                # Adicionar codificação e rótulo à lista de conhecidos
                known_encodings.append(face_encoding)
                known_labels.append(os.path.splitext(image_file)[0])  # Usa o nome do arquivo como rótulo

        frame_skip = 10
        frame_count = 0

        while True:
            ret, frame = video_capture.read()

            if not ret:
                break
    
            frame_count += 1

            # Pular frames se necessário
            if frame_count % frame_skip != 0:
                continue
    
            # Converter para escala de cinza  
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Obter localizações das faces no frame
            face_locations = face_recognition.face_locations(frame)
            for face_location in face_locations:
                # Obter coordenadas da caixa delimitadora
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

            cv2.imshow("Video", frame)
    
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        video_capture.release()
        cv2.destroyAllWindows()
        messagebox.showinfo("Sucesso", "O programa foi executado com sucesso!")
    except Exception as e:
        messagebox.showerror("Erro", f"Ocorreu um erro: {str(e)}")
        video_path.set("")
        dataset_path.set("")
        video_entry.delete(0, tk.END)
        dataset_entry.delete(0, tk.END)

# Criando os widgets da interface
title_label = tk.Label(window, text="Titulo Qualquer", font=("Arial", 16))
video_label = tk.Label(window, text="Caminho do video:")
video_entry = tk.Entry(window, textvariable=video_path)
video_button = tk.Button(window, text="Selecionar", command=select_video)
dataset_label = tk.Label(window, text="Caminho do dataset:")
dataset_entry = tk.Entry(window, textvariable=dataset_path)
dataset_button = tk.Button(window, text="Selecionar", command=select_dataset)
send_button = tk.Button(window, text="Enviar", command=send_paths)

# Posicionando os widgets na interface
title_label.pack(pady=10)
video_label.pack()
video_entry.pack(fill=tk.X, padx=10)
video_button.pack(pady=5)
dataset_label.pack()
dataset_entry.pack(fill=tk.X, padx=10)
dataset_button.pack(pady=5)
send_button.pack(pady=10)

# Iniciando o loop principal da interface
window.mainloop()
