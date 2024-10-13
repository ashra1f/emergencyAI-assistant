import cv2
import base64
from mistralai import Mistral

# Remplacer votre clé API ici
api_key = "eKf96BlAYMXLmBj2VJHkP443jrzXEZT2"

# Définir le modèle à utiliser (en supposant que 'pixtral-12b-2409' soit correct)
model = "pixtral-12b-2409"  # Assurez-vous que c'est la bonne version du modèle multimodal

# Initialiser le client Mistral avec la clé API
client = Mistral(api_key=api_key)

# Fonction pour envoyer l'image au modèle multimodal
def describe_image_with_multimodal(image_path):
    try:
        # Lire l'image en utilisant OpenCV
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("L'image n'a pas pu être lue, vérifiez le chemin.")

        # Encoder l'image au format JPEG
        success, encoded_image = cv2.imencode('.jpg', image)
        if not success:
            raise ValueError("Échec de l'encodage de l'image.")

        # Convertir l'image en base64 et ajouter le préfixe approprié
        image_data = base64.b64encode(encoded_image).decode('utf-8')
        image_base64_url = f"data:image/jpeg;base64,{image_data}"

        # Préparer les données pour envoyer au modèle multimodal
        chat_response = client.chat.complete(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text":'''you are a first responder and you are trying to know what is wrong with the person on the picture try to discribe the symptoms what he is suffering from and describe the syptoms in details '''
                        },
                        {
                            "type": "image_url",  # Garder le type 'image_url'
                            "image_url": image_base64_url  # Envoyer l'image encodée en base64 sous forme de data URL
                        }
                    ]
                },
            ]
        )

        # Extraire et retourner la réponse du modèle (en supposant qu'elle soit accessible via dot notation)
        return chat_response.choices[0].message.content

    except Exception as e:
        return f"Une erreur s'est produite : {str(e)}"

# Demander le chemin de l'image à l'utilisateur
image_path = input("Veuillez entrer le chemin de l'image : ")

# Tester la fonction avec le chemin de l'image fourni par l'utilisateur
image_description = describe_image_with_multimodal(image_path)

# Afficher la description générée par l'IA de l'image
print(image_description)
