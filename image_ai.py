import tkinter as tk
from tkinter import filedialog
import cv2
import base64
from PIL import Image, ImageTk, ImageDraw
from mistralai import Mistral
import whisper
import pyaudio
import wave
import threading
import os

# Replace with your API key
api_key = "eKf96BlAYMXLmBj2VJHkP443jrzXEZT2"
model = "pixtral-12b-2409"
client = Mistral(api_key=api_key)

# Class for recording and transcribing audio using Whisper
class WhisperApp:
    def __init__(self, root):
        self.is_recording = False
        self.filename = "output.wav"
        self.recording_thread = None
        self.root = root  # Store reference to the root window for after callbacks
        
        # Create the "Start/Stop Recording" button
        self.record_button = self.create_button(root, "Start Recording", self.toggle_recording, bg_color="#E74C3C")

    def create_button(self, parent, text, command, bg_color):
        """Helper method to create styled buttons."""
        return tk.Button(
            parent,
            text=text,
            command=command,
            width=15,
            height=2,
            bg=bg_color,
            fg="white",
            activebackground="#C0392B",
            activeforeground="white",
            font=("Helvetica", 10, "bold"),
            relief="flat",
            bd=2,
            highlightthickness=0
        )

    def toggle_recording(self):
        if self.is_recording:
            self.stop_recording()
        else:
            self.start_recording()

    def start_recording(self):
        self.is_recording = True
        self.record_button.config(text="Stop Recording")

        # Start recording in a separate thread
        self.recording_thread = threading.Thread(target=self.record_audio)
        self.recording_thread.start()

    def stop_recording(self):
        self.is_recording = False
        if self.recording_thread:
            self.recording_thread.join()  # Ensure the recording thread finishes
        self.record_button.config(text="Start Recording")

    def record_audio(self):
        chunk = 1024  # Record in chunks of 1024 samples
        sample_format = pyaudio.paInt16  # 16 bits per sample
        channels = 1
        fs = 44100  # Record at 44100 samples per second

        p = pyaudio.PyAudio()

        stream = p.open(format=sample_format,
                        channels=channels,
                        rate=fs,
                        frames_per_buffer=chunk,
                        input=True)

        print("Recording...")
        frames = []  # Initialize array to store frames

        while self.is_recording:
            data = stream.read(chunk)
            frames.append(data)

        # Stop and close the stream
        stream.stop_stream()
        stream.close()
        p.terminate()

        print("Finished recording.")

        # Save the recorded data as a WAV file
        wf = wave.open(self.filename, 'wb')
        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(sample_format))
        wf.setframerate(fs)
        wf.writeframes(b''.join(frames))
        wf.close()

        # Start transcription after recording is stopped
        self.transcribe_audio()

    def transcribe_audio(self):
        transcription_thread = threading.Thread(target=self.run_whisper)
        transcription_thread.start()

    def run_whisper(self):
        model = whisper.load_model("medium")
        result = model.transcribe(self.filename)

        # Update the text box with the transcribed text (thread-safe way)
        self.root.after(0, lambda: input_text.delete(1.0, tk.END))  # Clear any existing text
        self.root.after(0, lambda: input_text.insert(tk.END, result['text']))  # Insert the transcribed text

        os.remove(self.filename)  # Delete the audio file after transcription to save space

# Function to display the webcam feed in the Tkinter window
def open_camera():
    global cap, camera_window, camera_label

    # Create a new window for the camera
    camera_window = tk.Toplevel()
    camera_window.title("Camera Feed")
    camera_window.geometry("800x600")  # Set a larger window size for better visibility
    camera_window.configure(bg="#F7F7F7")

    # Create the label to display the camera feed in the new window
    camera_label = tk.Label(camera_window, bg="#333333", bd=5, relief="solid")
    camera_label.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

    # Initialize the camera
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        add_message_to_chat("Error: Unable to access the webcam.")
        return

    update_frame(60)  # Update the frame every 60ms

    # Bind a click event on the camera feed to capture the image
    camera_label.bind("<Button-1>", capture_image)

# Function to update the camera feed in the new window
def update_frame(interval):
    ret, frame = cap.read()
    if ret:
        # Resize the frame for better visibility in the large window
        frame = cv2.resize(frame, (800, 600))

        # Convert the OpenCV frame (BGR) to Tkinter image (RGB)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)

        # Draw a white dot (simulating a capture button) on the camera feed
        draw = ImageDraw.Draw(img)
        dot_position = (int(img.width / 2) - 20, img.height - 60)  # White dot position
        dot_size = 40  # Size of the dot
        draw.ellipse([dot_position[0], dot_position[1], dot_position[0] + dot_size, dot_position[1] + dot_size], fill="white")

        imgtk = ImageTk.PhotoImage(image=img)

        # Display the image in the camera_label
        camera_label.imgtk = imgtk
        camera_label.config(image=imgtk)

    # Continue updating the camera feed
    camera_label.after(interval, lambda: update_frame(interval))

# Function to capture an image from the webcam, and immediately close the camera
def capture_image(event):
    ret, frame = cap.read()
    if ret:
        cv2.imwrite("captured_image.jpg", frame)
        add_message_to_chat("Image captured and saved as 'captured_image.jpg'.", "user")
        describe_image("captured_image.jpg")

    close_camera()  # Close the camera window immediately after capturing the image

# Function to close the camera and the window after capturing
def close_camera():
    cap.release()  # Release the camera
    camera_label.config(image='')  # Clear the displayed image
    camera_window.destroy()  # Close the camera window
    cv2.destroyAllWindows()  # Close all OpenCV windows

# Function to describe an image using the multimodal model
def describe_image(image_path):
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Error: Unable to read the image.")

        success, encoded_image = cv2.imencode('.jpg', image)
        if not success:
            raise ValueError("Error: Failed to encode the image.")

        image_data = base64.b64encode(encoded_image).decode('utf-8')
        image_base64_url = f"data:image/jpeg;base64,{image_data}"

        # Send request to the multimodal model
        chat_response = client.chat.complete(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": '''Project Context:*
You are developing an AI assistant focused on first aid. This assistant must be capable of transforming image descriptions depicting emergency situations into relevant questions. These questions will then be submitted to another AI model specialized in emergency cases to get recommendations on the appropriate actions to take.

*Prompt Objective:*
Rephrase an image description related to first aid into a clear, actionable question, intended for use by another AI model specialized in emergency scenarios, to provide suitable instructions.

*Instructions:*

1. *Contextual Analysis:* Understand the emergency situation described in the image.
2. *Rephrasing into a Question:* Transform the description into a clear, relevant, action-oriented question.
3. *Relevance and Actionability:* Ensure that the question elicits specific and immediate recommendations in the context of first aid.
4. *Inter-Model Interaction:* Generate a question that will be passed to another AI model specialized in emergencies to provide action steps.

*Examples:*

---

Image Description: A man is lying on a stretcher in an emergency room with a knife lodged in his abdomen.

Generated Question:
What first aid actions should an AI assistant specialized in emergency care recommend for a man lying on a stretcher with a knife lodged in his abdomen in an emergency room?

---

Image Description: A woman is unconscious on the ground in a park after being hit by a bicycle.

Generated Question:
What first aid steps should an AI assistant specialized in emergencies recommend for handling an unconscious woman after a bicycle accident in a park?

---

Image Description: A child is choking on an apple lodged in their throat, surrounded by concerned witnesses.

Generated Question:
How should an AI assistant specialized in first aid guide witnesses to help a child choking on an apple?

---

Image Description: A person performing CPR on a colleague who suffered a cardiac arrest in an office setting.

Generated Question:
What CPR steps should an AI assistant specialized in emergency care recommend during a cardiac arrest in an office environment?

---

Image Description: An individual has sustained third-degree burns from a house fire, requiring urgent medical intervention.

Generated Question:
What first aid measures should an AI assistant specialized in emergencies recommend for treating third-degree burns caused by a house fire?

---

Image Description: A man has broken his leg after a fall while hiking in the mountains and is immobilized.

Generated Question:
What first aid actions should an AI assistant recommend for managing a broken leg after a fall during a mountain hike?

---

Image Description: A person is experiencing a severe allergic reaction after consuming peanuts, showing signs of anaphylactic shock in a classroom.

Generated Question:
What interventions should an AI assistant specialized in first aid recommend for treating anaphylactic shock caused by a peanut allergy in a classroom?

---

Image Description: A drowning victim is partially out of the water, awaiting help on the beach.

Generated Question:
What first aid actions should an AI assistant recommend to assist a partially drowned person while waiting for emergency services on the beach?

---

Image Description: A person is suffering from severe food poisoning, showing symptoms of vomiting and abdominal cramps in a restaurant.

Generated Question:
What first aid measures should an AI assistant recommend for managing severe food poisoning with vomiting and abdominal cramps?

---

Image Description: A cyclist is severely injured after a road accident, with multiple injuries requiring urgent medical evacuation.

Generated Question:
What are the priority actions an AI assistant should recommend for managing a severely injured cyclist after a road accident?

---

Image Description: A person is having a seizure in public, surrounded by confused bystanders.

Generated Question:
What first aid advice should an AI assistant provide for managing a seizure in public and assisting disoriented bystanders?

---

Image Description: A man is experiencing a heart attack in the middle of a busy street, with passersby attempting to assist.

Generated Question:
What immediate steps should an AI assistant recommend to assist a man having a heart attack in a busy street with helping passersby?

---

Image Description: A person has sustained a severe ankle sprain after slipping on ice, with witnesses trying to stabilize the injury.

Generated Question:
What first aid actions should an AI assistant recommend for managing a severe ankle sprain after a fall on ice, with witnesses trying to help?

---

Image Description: A baby is showing signs of choking during feeding, with panicked parents seeking help.

Generated Question:
How should an AI assistant guide parents to help a baby choking during feeding?

---

Image Description: A person is suffering from carbon monoxide poisoning in an apartment, showing symptoms of confusion and weakness.

Generated Question:
What first aid measures should an AI assistant recommend for treating carbon monoxide poisoning with symptoms of confusion and weakness in an apartment?

---

Image Description: An athlete collapses on the field after a cardiac episode, with teammates trying to assist.

Generated Question:
What first aid actions should an AI assistant recommend to assist an athlete after a cardiac event on the field, with teammates trying to help?

---

Image Description: A person hits their head against a wall due to a fall, showing signs of a concussion in a public space.

Generated Question:
What interventions should an AI assistant recommend for managing a concussion after a person hits their head in a public space?

---

Image Description: A person is bitten by a venomous snake in a rural area, showing signs of shock and intense pain.

Generated Question:
What priority actions should an AI assistant recommend for treating a venomous snake bite in a rural area with signs of shock and intense pain?

---

Image Description: A person is having a stroke, displaying signs of one-sided paralysis in a workplace.

Generated Question:
What first aid measures should an AI assistant recommend for managing a stroke with one-sided paralysis in a workplace?

---

*Prompt Usage Instructions:*

1. *Replace the Description:* Insert the new image description where indicated.
2. *Generate the Question:* Let the AI assistant generate the relevant question based on the provided examples.
3. *Verify:* Ensure that the generated question is clear, relevant, action-oriented, and adheres to first aid protocols.
'''
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
        add_message_to_chat(f"AI Assistant: {description}", "ai")

    except Exception as e:
        add_message_to_chat(f"Error: {str(e)}", "ai")

# Function to send text input to the API
def send_text():
    user_input = input_text.get("1.0", tk.END).strip()
    if not user_input:
        add_message_to_chat("Error: No text entered.", "user")
        return

    add_message_to_chat(f"User: {user_input}", "user")

    try:
        # Send request to the model with the user input
        chat_response = client.chat.complete(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": user_input
                }
            ]
        )

        response = chat_response.choices[0].message.content
        add_message_to_chat(f"QdAI: {response}", "ai")

    except Exception as e:
        add_message_to_chat(f"Error: {str(e)}", "ai")

    # Clear the input text box after sending the message
    input_text.delete(1.0, tk.END)

# Function to upload an image
def upload_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    if file_path:
        add_message_to_chat(f"Image uploaded: {file_path}", "user")
        describe_image(file_path)

# Function to add a message to the chat window
def add_message_to_chat(message, sender):
    chat_text.config(state=tk.NORMAL)  # Unlock the chat text box for editing
    
    # Apply formatting based on whether the sender is "user" or "ai"
    if sender == "user":
        chat_text.insert(tk.END, f"{message}\n", "user")  # Append the new message
    elif sender == "ai":
        chat_text.insert(tk.END, f"{message}\n", "ai")  # Append AI message in bold

    chat_text.config(state=tk.DISABLED)  # Lock the chat text box after editing
    chat_text.see(tk.END)  # Scroll to the bottom of the chat box

# Tkinter GUI setup
def create_gui():
    global window, camera_label, chat_text, input_text

    window = tk.Tk()
    window.title("QuickAid")
    window.geometry("450x800")  # Adjusted window size for a mobile-like appearance
    window.configure(bg="#F7F7F7")  # Set background to light gray

    # Chat window to display messages like a chatbot conversation
    chat_text = tk.Text(window, height=20, width=50, bg="white", fg="black", wrap="word", bd=2, relief="solid", font=("Arial", 10))
    chat_text.pack(pady=10, padx=10)
    chat_text.config(state=tk.DISABLED)  # Disable direct editing of the chat

    # Configure tags for the chat text formatting
    chat_text.tag_configure("user", foreground="gray")  # User messages in gray
    chat_text.tag_configure("ai", font=("Arial", 10, "bold"))  # AI messages in bold

    # Text input field for typing messages
    input_text = tk.Text(window, height=3, width=40, bg="#FFFFFF", fg="black", bd=2, relief="solid", font=("Arial", 10))
    input_text.pack(pady=10, padx=10)

    # Button frame (for horizontal button layout)
    button_frame_top = tk.Frame(window, bg="#F7F7F7")
    button_frame_top.pack(pady=10)

    # First row of buttons: Send Text and Start Recording
    send_text_btn = WhisperApp(window).create_button(button_frame_top, "Send Text", send_text, bg_color="#3498DB")
    send_text_btn.pack(side=tk.LEFT, padx=5)

    whisper_app = WhisperApp(button_frame_top)
    whisper_app.record_button.pack(side=tk.LEFT, padx=5)

    # Second row of buttons: Open Camera and Upload Image
    button_frame_bottom = tk.Frame(window, bg="#F7F7F7")
    button_frame_bottom.pack(pady=10)

    open_camera_btn = WhisperApp(window).create_button(button_frame_bottom, "Open Camera", open_camera, bg_color="#2ECC71")
    open_camera_btn.pack(side=tk.LEFT, padx=5)

    upload_image_btn = WhisperApp(window).create_button(button_frame_bottom, "Upload Image", upload_image, bg_color="#F39C12")
    upload_image_btn.pack(side=tk.LEFT, padx=5)

    window.mainloop()

if __name__ == "__main__":
    create_gui()
