import whisper

def transcribe_video(file_path, model_size="medium"):
    """
    Transcribe an audio or video file using Whisper.
    
    Args:
    - file_path (str): Path to the audio or video file.
    - model_size (str): Size of the Whisper model to load. Options are 'base', 'small', 'medium', 'large'.
    
    Returns:
    - str: The transcribed text.
    """
    # Load the Whisper model inside the function
    model = whisper.load_model(model_size)
    
    # Transcribe the file
    result = model.transcribe(file_path)
    
    # Return the transcribed text
    return result['text']

# Example usage:
file_path = "C:\\Users\\Stagiaire\\Desktop\\farouko\\Yass.mp4"
transcription = transcribe_video(file_path, model_size="medium")
print(transcription)
