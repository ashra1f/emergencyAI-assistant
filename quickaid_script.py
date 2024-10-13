import os
import jsonlines
import tkinter as tk
from tkinter import filedialog
import cv2
import base64
from PIL import Image, ImageTk, ImageDraw
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from mistralai import Mistral  # Assurez-vous d'avoir le bon client pour Mistral
import whisper
import pyaudio
import wave
import threading

# Configuration des clés API et des modèles
API_KEY_MISTRAL = "5PWT0UWdQuVg4kQAetY6SOi6SdgVLisn"  # Remplacez par votre clé API Mistral
RAG_MODEL_NAME = "open-mistral-7b"  # Assurez-vous que ce modèle est disponible
MULTIMODAL_MODEL_NAME = "pixtral-12b-2409"  # Modèle pour la description d'image

# Initialisation du client Mistral
client = Mistral(api_key=API_KEY_MISTRAL)

# Fonction RAG avec le nouveau prompt intégré
def ask_question_rag(question, context, embedding_model, index, documents):
    prompt = (
        f"Context: {context}\n\n"
        f"Medical Question: {question}\n\n"
        "You will receive a medical question as input. Analyze the question carefully to extract clear instructions and identify the primary action or response needed. "
        "Provide a concise, direct solution that is medically relevant, actionable, and avoids any ambiguity. "
        "Ensure the response is precise and adheres to medical best practices.\nAnswer:"
    )
    chat_response = client.chat.complete(
        model=RAG_MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": prompt,
            },
        ]
    )
    return chat_response.choices[0].message.content

# Fonction pour charger les documents JSONL
def load_documents(jsonl_path):
    documents = []
    with jsonlines.open(jsonl_path) as reader:
        for obj in reader:
            prompt = obj.get('prompt', '')
            completion = obj.get('completion', '')
            document = f"Prompt: {prompt}\nCompletion: {completion}"
            documents.append(document)
    return documents

# Fonction pour créer les embeddings
def create_embeddings(documents, embedding_model):
    embeddings = embedding_model.encode(documents, convert_to_numpy=True, show_progress_bar=True)
    return embeddings

# Fonction pour créer l'index FAISS
def create_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

# Classe pour l'application Whisper
class WhisperApp:
    def __init__(self, root, rag_system):
        self.is_recording = False
        self.filename = "output.wav"
        self.recording_thread = None
        self.root = root
        self.rag_system = rag_system  # Référence au système RAG

        # Créer le bouton d'enregistrement en cercle
        self.record_button = self.create_circle_button(root, 50, 50, 40, "Start", self.toggle_recording, "#E74C3C")

    def create_circle_button(self, parent, x, y, r, text, command, color):
        """Crée un bouton circulaire avec du texte à l'intérieur."""
        canvas = tk.Canvas(parent, width=2*r, height=2*r, bg="#F7F7F7", highlightthickness=0)
        canvas.pack(pady=10)  # Ajuster l'espacement
        canvas.create_oval(x-r, y-r, x+r, y+r, fill=color, outline="")  # Cercle

        # Créer le texte à l'intérieur du cercle
        canvas.create_text(x, y, text=text, fill="white", font=("Helvetica", 10, "bold"))

        # Ajouter un bind pour gérer le clic comme un bouton
        canvas.bind("<Button-1>", lambda event: command())
        return canvas

    def toggle_recording(self):
        if self.is_recording:
            self.stop_recording()
        else:
            self.start_recording()

    def start_recording(self):
        self.is_recording = True
        self.record_button.delete("all")
        self.create_circle_button(self.root, 50, 50, 40, "Stop", self.toggle_recording, "#C0392B")
        # Démarrer l'enregistrement dans un thread séparé
        self.recording_thread = threading.Thread(target=self.record_audio)
        self.recording_thread.start()

    def stop_recording(self):
        self.is_recording = False
        if self.recording_thread:
            self.recording_thread.join()  # S'assurer que le thread d'enregistrement se termine
        self.record_button.delete("all")
        self.create_circle_button(self.root, 50, 50, 40, "Start", self.toggle_recording, "#E74C3C")

    def record_audio(self):
        chunk = 1024  # Enregistrer par blocs de 1024 échantillons
        sample_format = pyaudio.paInt16  # 16 bits par échantillon
        channels = 1
        fs = 44100  # Enregistrer à 44100 échantillons par seconde

        p = pyaudio.PyAudio()

        stream = p.open(format=sample_format,
                        channels=channels,
                        rate=fs,
                        frames_per_buffer=chunk,
                        input=True)

        print("Recording...")
        frames = []  # Initialiser le tableau pour stocker les frames

        while self.is_recording:
            data = stream.read(chunk)
            frames.append(data)

        # Arrêter et fermer le stream
        stream.stop_stream()
        stream.close()
        p.terminate()

        print("Finished recording.")

        # Sauvegarder les données enregistrées en tant que fichier WAV
        wf = wave.open(self.filename, 'wb')
        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(sample_format))
        wf.setframerate(fs)
        wf.writeframes(b''.join(frames))
        wf.close()

        # Démarrer la transcription après l'arrêt de l'enregistrement
        self.transcribe_audio()

    def transcribe_audio(self):
        transcription_thread = threading.Thread(target=self.run_whisper)
        transcription_thread.start()

    def run_whisper(self):
        model = whisper.load_model("medium")
        result = model.transcribe(self.filename)

        # Mettre à jour la zone de texte avec le texte transcrit (méthode thread-safe)
        self.root.after(0, lambda: input_text.delete(1.0, tk.END))  # Effacer le texte existant
        self.root.after(0, lambda: input_text.insert(tk.END, result['text']))  # Insérer le texte transcrit

        os.remove(self.filename)  # Supprimer le fichier audio après la transcription pour économiser de l'espace

        # Utiliser la transcription comme contexte pour le système RAG
        self.rag_system.set_context(result['text'])

# Classe pour le système RAG intégré
class RAGSystem:
    def __init__(self, jsonl_path):
        print("Initialisation du système RAG...")
        self.documents = load_documents(jsonl_path)
        print("Chargement du modèle d'embedding...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # Vous pouvez choisir un autre modèle
        print("Création des embeddings...")
        self.embeddings = create_embeddings(self.documents, self.embedding_model)
        print("Indexation avec FAISS...")
        self.index = create_faiss_index(self.embeddings)
        self.context = ""  # Contexte initial vide

    def set_context(self, context):
        self.context = context
        print(f"Contexte mis à jour: {self.context}")

    def get_answer(self, question):
        if not self.context:
            return "Contexte non défini. Veuillez fournir un contexte en décrivant une image ou en saisissant du texte."
        answer = ask_question_rag(question, self.context, self.embedding_model, self.index, self.documents)
        return answer

# Classe pour l'application Tkinter avec format iPhone
class QuickAidApp:
    def __init__(self, root):
        self.root = root
        self.root.title("QuickAid")
        self.root.geometry("390x844")  # Format portrait pour un iPhone-like format
        self.root.configure(bg="#F7F7F7")  # Fond gris clair

        # Initialiser le système RAG
        self.rag_system = RAGSystem("data.jsonl")

        # Initialiser Whisper
        self.whisper_app = WhisperApp(root, self.rag_system)

        # Fenêtre de chat pour afficher les messages
        self.chat_text = tk.Text(root, height=20, width=35, bg="white", fg="black", wrap="word", bd=2, relief="solid", font=("Arial", 10))
        self.chat_text.pack(pady=10, padx=10, fill=tk.BOTH)
        self.chat_text.config(state=tk.DISABLED)  # Désactiver l'édition directe du chat

        # Configurer les tags pour le formatage du chat
        self.chat_text.tag_configure("user", foreground="blue")  # Messages de l'utilisateur en bleu
        self.chat_text.tag_configure("ai", foreground="green", font=("Arial", 10, "bold"))  # Messages de l'AI en vert et gras

        # Zone de saisie de texte pour taper des messages
        self.input_text = tk.Text(root, height=3, width=35, bg="#FFFFFF", fg="black", bd=2, relief="solid", font=("Arial", 10))
        self.input_text.pack(pady=5, padx=10, fill=tk.X)

        # Cadre pour organiser les boutons de manière verticale
        button_frame = tk.Frame(root, bg="#F7F7F7")
        button_frame.pack(pady=5)

        # Bouton "Start Recording" de WhisperApp (ligne supérieure)
        self.whisper_app.record_button.pack(pady=5, fill=tk.X)

        # Bouton "Send Text" (sous le bouton "Start Recording")
        send_text_btn = self.create_button(button_frame, "Send Text", self.send_text, bg_color="#3498DB")
        send_text_btn.pack(pady=5, fill=tk.X)

        # Bouton "Open Camera" (sous "Send Text")
        open_camera_btn = self.create_button(button_frame, "Open Camera", self.open_camera, bg_color="#2ECC71")
        open_camera_btn.pack(pady=5, fill=tk.X)

        # Bouton "Upload Image" (sous "Open Camera")
        upload_image_btn = self.create_button(button_frame, "Upload Image", self.upload_image, bg_color="#F39C12")
        upload_image_btn.pack(pady=5, fill=tk.X)

    def create_button(self, parent, text, command, bg_color):
        """Méthode d'aide pour créer des boutons stylés."""
        return tk.Button(
            parent,
            text=text,
            command=command,
            width=15,  # Taille ajustée pour l'écran plus petit
            height=2,  # Hauteur des boutons réduite
            bg=bg_color,
            fg="white",
            activebackground="#C0392B",
            activeforeground="white",
            font=("Helvetica", 10, "bold"),
            relief="flat",
            bd=2,
            highlightthickness=0
        )

    def send_text(self):
        user_input = self.input_text.get("1.0", tk.END).strip()
        if not user_input:
            self.add_message_to_chat("Erreur : Aucun texte saisi.", "user")
            return

        self.add_message_to_chat(f"Utilisateur: {user_input}", "user")

        try:
            # Obtenir la réponse du système RAG
            response = self.rag_system.get_answer(user_input)
            self.add_message_to_chat(f"{response}", "ai")  # Afficher seulement les instructions
        except Exception as e:
            self.add_message_to_chat(f"Erreur : {str(e)}", "ai")

        # Effacer la zone de saisie de texte après envoi
        self.input_text.delete(1.0, tk.END)

    # Fonction pour ouvrir la caméra
    def open_camera(self):
        global cap, camera_window, camera_label

        # Créer une nouvelle fenêtre pour la caméra
        camera_window = tk.Toplevel()
        camera_window.title("Camera Feed")
        camera_window.geometry("390x300")  # Taille ajustée pour format portrait
        camera_window.configure(bg="#F7F7F7")

        # Créer le label pour afficher le flux de la caméra
        camera_label = tk.Label(camera_window, bg="#333333", bd=5, relief="solid")
        camera_label.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

        # Initialiser la caméra
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            self.add_message_to_chat("Erreur : Impossible d'accéder à la webcam.", "ai")
            return

        self.update_frame(60)  # Mettre à jour le frame toutes les 60ms

        # Lier un événement de clic sur le flux de la caméra pour capturer l'image
        camera_label.bind("<Button-1>", self.capture_image)

    def update_frame(self, interval):
        ret, frame = cap.read()
        if ret:
            # Redimensionner le frame pour une meilleure visibilité dans la petite fenêtre
            frame = cv2.resize(frame, (390, 300))

            # Convertir le frame OpenCV (BGR) en image Tkinter (RGB)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)

            # Dessiner un point blanc (simulant un bouton de capture) sur le flux de la caméra
            draw = ImageDraw.Draw(img)
            dot_position = (int(img.width / 2) - 20, img.height - 60)  # Position du point blanc
            dot_size = 40  # Taille du point
            draw.ellipse([dot_position[0], dot_position[1], dot_position[0] + dot_size, dot_position[1] + dot_size], fill="white")

            imgtk = ImageTk.PhotoImage(image=img)

            # Afficher l'image dans le camera_label
            camera_label.imgtk = imgtk
            camera_label.config(image=imgtk)

        # Continuer à mettre à jour le flux de la caméra
        camera_label.after(interval, lambda: self.update_frame(interval))

    def capture_image(self, event):
        ret, frame = cap.read()
        if ret:
            cv2.imwrite("captured_image.jpg", frame)
            self.add_message_to_chat("Image capturée et enregistrée sous 'captured_image.jpg'.", "user")
            self.describe_image("captured_image.jpg")

        self.close_camera()  # Fermer la fenêtre de la caméra immédiatement après la capture

    def close_camera(self):
        cap.release()  # Libérer la caméra
        camera_label.config(image='')  # Effacer l'image affichée
        camera_window.destroy()  # Fermer la fenêtre de la caméra
        cv2.destroyAllWindows()  # Fermer toutes les fenêtres OpenCV

    def describe_image(self, image_path):
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError("Erreur : Impossible de lire l'image.")

            success, encoded_image = cv2.imencode('.jpg', image)
            if not success:
                raise ValueError("Erreur : Échec de l'encodage de l'image.")

            image_data = base64.b64encode(encoded_image).decode('utf-8')
            image_base64_url = f"data:image/jpeg;base64,{image_data}"

            # Envoyer la requête au modèle multimodal
            chat_response = client.chat.complete(
                model=MULTIMODAL_MODEL_NAME,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "You are a first responder tasked with analyzing visual input to assist in medical emergencies. Your role is to carefully observe the provided images of individuals in distress, identify any visible symptoms in detail, such as physical injuries, signs of illness, or abnormal conditions. Based on your observations, formulate a medical first aid question that addresses the possible condition.."
                            },
                            {
                                "type": "image_url",
                                "image_url": image_base64_url
                            }
                        ]
                    },
                ]
            )

            description = chat_response.choices[0].message.content
            self.add_message_to_chat(f"{description}", "ai")  # Afficher seulement la description

            # Utiliser la description comme contexte pour RAG
            self.rag_system.set_context(description)

            # Créer un prompt spécifique pour RAG basé sur la description
            prompt = f"Based on the following description: '{description}', please provide the necessary instructions for a first responder."
            
            # Poser la question à RAG avec ce prompt
            instructions = self.rag_system.get_answer(prompt)
            self.add_message_to_chat(f"{instructions}", "ai")  # Afficher seulement les instructions

        except Exception as e:
            self.add_message_to_chat(f"Erreur : {str(e)}", "ai")

    def upload_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", ".jpg;.jpeg;*.png")])
        if file_path:
            self.add_message_to_chat(f"Image téléchargée : {file_path}", "user")
            self.describe_image(file_path)

    def add_message_to_chat(self, message, sender):
        self.chat_text.config(state=tk.NORMAL)  # Déverrouiller la zone de chat pour édition

        if sender == "user":
            self.chat_text.insert(tk.END, f"{message}\n", "user")  # Ajouter le message utilisateur
        elif sender == "ai":
            self.chat_text.insert(tk.END, f"{message}\n", "ai")  # Ajouter seulement les instructions

        self.chat_text.config(state=tk.DISABLED)  # Verrouiller la zone de chat après édition
        self.chat_text.see(tk.END)  # Faire défiler jusqu'en bas de la zone de chat

# Fonction principale pour créer l'interface graphique
def create_gui():
    window = tk.Tk()
    app = QuickAidApp(window)
    window.mainloop()

if __name__ == "__main__":
    create_gui()
