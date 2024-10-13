import tkinter as tk
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
        self.root = root
        self.is_recording = False
        self.filename = "output.wav"
        self.recording_thread = None
        
        # Create the "Start/Stop Recording" round button
        self.record_button = self.create_round_button(root, "Start Recording", self.toggle_recording, button_color="red")
        self.record_button.pack(pady=10)

    # Create a round button using Canvas
    def create_round_button(self, parent, text, command, button_color="red"):
        button_frame = tk.Frame(parent, bg="white")
        canvas = tk.Canvas(button_frame, width=100, height=100, highlightthickness=0, bg="white")
        canvas.pack()

        # Create a circle (button shape)
        circle = canvas.create_oval(10, 10, 90, 90, outline=button_color, fill=button_color)
        
        # Add button text inside the circle
        button_text = canvas.create_text(50, 50, text=text, fill="white", font=('Helvetica', 9, 'bold'))
        
        # Bind the click event to the circle
        canvas.tag_bind(circle, "<Button-1>", lambda event: command())
        canvas.tag_bind(button_text, "<Button-1>", lambda event: command())

        return button_frame

    # Toggle the recording (start/stop with the same button)
    def toggle_recording(self):
        if self.is_recording:
            self.stop_recording()
        else:
            self.start_recording()

    def start_recording(self):
        self.is_recording = True
        self.record_button.winfo_children()[0].itemconfig(2, text="Stop Recording")  # Update button text
        
        # Start recording in a separate thread
        self.recording_thread = threading.Thread(target=self.record_audio)
        self.recording_thread.start()

    def stop_recording(self):
        self.is_recording = False
        if self.recording_thread:
            self.recording_thread.join()  # Ensure the recording thread finishes before proceeding
        self.record_button.winfo_children()[0].itemconfig(2, text="Start Recording")  # Update button text

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

        # Insert transcribed text into the input_text field
        input_text.delete(1.0, tk.END)  # Clear existing text
        input_text.insert(tk.END, result['text'])  # Insert transcribed text

        os.remove(self.filename)  # Delete the audio file after transcription to save space

# Function to display the webcam feed in the Tkinter window
def open_camera():
    global cap
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        add_message_to_chat("Error: Unable to access the webcam.")
        return

    # Optimize the frame update interval for better performance
    update_frame(60)  # Updating the frame every 60ms instead of 10ms

# Function to update the webcam feed with a round button (white dot)
def update_frame(interval):
    ret, frame = cap.read()
    if ret:
        # Convert the OpenCV frame (BGR) to Tkinter image (RGB)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)

        # Draw a white dot (simulating a round button) on the camera feed
        draw = ImageDraw.Draw(img)
        button_position = (int(img.width/2) - 15, img.height - 50)  # White dot position
        button_size = 30  # Size of the dot
        draw.ellipse([button_position[0], button_position[1], button_position[0] + button_size, button_position[1] + button_size], fill="white")

        imgtk = ImageTk.PhotoImage(image=img)

        # Display the image in the camera_label
        camera_label.imgtk = imgtk
        camera_label.config(image=imgtk)

    # Adjust frame update rate to reduce processing load
    camera_label.after(interval, lambda: update_frame(interval))

# Function to capture an image from the webcam, wait 3 seconds, and close the camera
def capture_image(event):
    ret, frame = cap.read()
    if ret:
        cv2.imwrite("captured_image.jpg", frame)
        add_message_to_chat("Image captured and saved as 'captured_image.jpg'.")
        describe_image("captured_image.jpg")

    # Wait 3 seconds before closing the camera feed
    window.after(3000, close_camera)

# Function to close the camera after a delay
def close_camera():
    cap.release()  # Release the camera
    camera_label.config(image='')  # Clear the displayed image
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
                            "text": '''You are a first responder. Describe the symptoms of the person in the picture.'''
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
        add_message_to_chat(f"Image description: {description}")

    except Exception as e:
        add_message_to_chat(f"Error: {str(e)}")

# Function to send text input to the API
def send_text():
    user_input = input_text.get("1.0", tk.END).strip()
    if not user_input:
        add_message_to_chat("Error: No text entered.")
        return

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
        add_message_to_chat(f"API response: {response}")

    except Exception as e:
        add_message_to_chat(f"Error: {str(e)}")

# Function to add a message to the chat window
def add_message_to_chat(message):
    chat_text.config(state=tk.NORMAL)  # Unlock the chat text box for editing
    chat_text.insert(tk.END, f"{message}\n")  # Append the new message
    chat_text.config(state=tk.DISABLED)  # Lock the chat text box after editing
    chat_text.see(tk.END)  # Scroll to the bottom of the chat box

# Tkinter GUI setup
def create_gui():
    global window, camera_label, chat_text, input_text

    window = tk.Tk()
    window.title("Medical Themed Application")
    window.geometry("600x650")  # Adjusted window size for a more compact layout
    window.configure(bg="white")  # Set white background for the Red Cross theme

    # Chat window
    chat_text = tk.Text(window, height=10, width=60, bg="white", fg="black", wrap="word", bd=5, relief="solid")
    chat_text.pack(pady=10)
    chat_text.config(state=tk.DISABLED)  # Disable direct editing of the chat

    # Camera feed display
    camera_label = tk.Label(window, bg="black", bd=5, relief="solid")
    camera_label.pack(pady=10)

    # Text input field
    input_text = tk.Text(window, height=3, width=50, bg="white", fg="black", bd=5, relief="solid")
    input_text.pack(pady=10)

    # Button to send the entered text
    send_text_btn = create_round_button(window, "Send Text", send_text)
    send_text_btn.pack(pady=5)

    # Button to open the camera
    open_camera_btn = create_round_button(window, "Open Camera", open_camera)
    open_camera_btn.pack(pady=5)

    # Bind a click event to capture an image inside the camera label
    camera_label.bind("<Button-1>", capture_image)

    # Integrate the WhisperApp into the main Tkinter interface
    whisper_app = WhisperApp(window)

    window.mainloop()

# Create a round button for use in the interface
def create_round_button(parent, text, command):
    button_frame = tk.Frame(parent, bg="white")
    canvas = tk.Canvas(button_frame, width=100, height=100, highlightthickness=0, bg="white")
    canvas.pack()

    # Draw a circle (round button)
    circle = canvas.create_oval(10, 10, 90, 90, outline="red", fill="red")
    button_text = canvas.create_text(50, 50, text=text, fill="white", font=('Helvetica', 9, 'bold'))

    # Bind click event to the circle and text
    canvas.tag_bind(circle, "<Button-1>", lambda event: command())
    canvas.tag_bind(button_text, "<Button-1>", lambda event: command())

    return button_frame

if __name__ == "__main__":
    create_gui()
