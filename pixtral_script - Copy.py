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
                            "text":'''Vous développez un assistant IA dédié aux premiers secours. Cet assistant doit être capable de transformer des descriptions d'images représentant des situations d'urgence en questions pertinentes. Ces questions seront ensuite soumises à un autre modèle d'intelligence artificielle spécialisé dans les cas d'urgence pour obtenir des recommandations sur les actions à entreprendre.

Objectif du Prompt :
Reformuler une description d'image relative aux premiers secours sous forme de question pertinente et actionnable, destinée à être utilisée par un autre modèle d'IA spécialisé dans les cas d'urgence pour fournir des instructions appropriées.

Instructions :
1. Analyse Contextuelle : Comprenez la situation d'urgence décrite dans l'image.
2. Reformulation en Question : Transformez la description en une question claire, pertinente et orientée vers l'action.
3. Pertinence et Actionnabilité : Assurez-vous que la question permet d'obtenir des recommandations précises et immédiates en situation de premiers secours.
4. Interaction Inter-Modèle : Génère une question destinée à être soumise à un autre modèle d'intelligence artificielle spécialisé dans les cas d'urgence, afin d'obtenir des recommandations sur les actions à entreprendre.

Exemples :

---

Description de l'image : Un homme est allongé sur une civière dans une salle d'urgence, avec un couteau enfoncé dans son ventre.

Question générée :
Quelles actions de premiers secours un assistant IA spécialisé devrait-il recommander pour assister un homme allongé sur une civière avec un couteau enfoncé dans son ventre dans une salle d'urgence ?

---

Description de l'image : Une femme est inconsciente sur le sol d'un parc après avoir été heurtée par un vélo.

Question générée :
Quelles sont les actions de premiers secours qu'un assistant IA spécialisé devrait recommander pour prendre en charge une femme inconsciente après un accident de vélo dans un parc ?

---

Description de l'image : Un enfant étouffant avec une pomme coincée dans la gorge, entouré de témoins inquiets.

Question générée :
Comment un assistant IA spécialisé en premiers secours devrait-il guider les témoins pour aider un enfant qui s'étouffe avec une pomme dans la gorge ?

---

Description de l'image : Une personne en plein effort de réanimation cardiorespiratoire (RCR) sur un collègue victime d'un arrêt cardiaque dans un bureau.

Question générée :
Quelles étapes de réanimation cardiorespiratoire (RCR) un assistant IA spécialisé devrait-il conseiller lors d'un arrêt cardiaque en milieu de bureau ?

---

Description de l'image : Un individu brûlé au troisième degré par un incendie domestique, nécessitant une intervention médicale urgente.

Question générée :
Quelles mesures de premiers secours un assistant IA spécialisé devrait-il recommander pour traiter une brûlure au troisième degré causée par un incendie domestique ?

---

Description de l'image : Un homme souffrant d'une fracture de jambe après une chute lors d'une randonnée en montagne, immobilisé sur place.

Question générée :
Quelles actions de premiers secours un assistant IA spécialisé devrait-il suggérer pour gérer une fracture de jambe suite à une chute en randonnée en montagne ?

---

Description de l'image : Une personne victime d'une allergie sévère après avoir ingéré des arachides, manifestant des signes de choc anaphylactique dans une salle de classe.

Question générée :
Quelles interventions un assistant IA spécialisé devrait-il recommander pour traiter un choc anaphylactique causé par une allergie aux arachides dans une salle de classe ?

---

Description de l'image : Une victime de noyade partiellement sortie de l'eau, en attente de secours sur la plage.

Question générée :
Quelles actions de premiers secours un assistant IA spécialisé devrait-il recommander pour assister une personne partiellement noyée en attendant les secours sur la plage ?

---

Description de l'image : Une personne victime d'une intoxication alimentaire sévère, présentant des vomissements et des crampes abdominales dans un restaurant.

Question générée :
Quelles mesures de premiers secours un assistant IA spécialisé devrait-il suggérer pour gérer une intoxication alimentaire sévère avec des symptômes de vomissements et de crampes abdominales ?

---

Description de l'image : Un cycliste grièvement blessé après un accident de la route, affichant des blessures multiples et nécessitant une évacuation médicale urgente.

Question générée :
Quelles sont les actions prioritaires que devrait recommander un assistant IA spécialisé pour prendre en charge un cycliste grièvement blessé après un accident de la route ?

---

Description de l'image : Une personne souffrant d'une crise d'épilepsie en public, entourée de spectateurs désorientés.

Question générée :
Quels conseils de premiers secours un assistant IA spécialisé devrait-il fournir pour gérer une crise d'épilepsie en public et aider les spectateurs désorientés ?

---

Description de l'image : Un homme a une crise cardiaque en plein milieu d'une rue animée, avec des passants tentant de lui porter assistance.

Question générée :
Quelles étapes immédiates un assistant IA spécialisé devrait-il recommander pour assister un homme en crise cardiaque dans une rue animée avec des passants aidants ?

---

Description de l'image : Une personne souffrant d'une entorse grave après avoir glissé sur une glace, entourée de témoins essayant de stabiliser la blessure.

Question générée :
Quelles actions de premiers secours un assistant IA spécialisé devrait-il recommander pour gérer une entorse grave suite à une chute sur glace, en présence de témoins aidants ?

---

Description de l'image : Un bébé présentant des signes de suffocation lors de l'alimentation, avec des parents paniqués cherchant de l'aide.

Question générée :
Comment un assistant IA spécialisé en premiers secours devrait-il guider les parents pour aider un bébé en situation de suffocation lors de l'alimentation ?

---

Description de l'image : Une personne victime d'une intoxication au monoxyde de carbone dans un appartement, présentant des symptômes de confusion et de faiblesse.

Question générée :
Quelles mesures de premiers secours un assistant IA spécialisé devrait-il recommander pour traiter une intoxication au monoxyde de carbone dans un appartement avec des symptômes de confusion et de faiblesse ?

---

Description de l'image : Un sportif s'effondre sur le terrain après un malaise cardiaque, avec des coéquipiers tentant de lui porter assistance.

Question générée :
Quelles actions de premiers secours un assistant IA spécialisé devrait-il recommander pour assister un sportif victime d'un malaise cardiaque sur le terrain, en présence de coéquipiers aidants ?

---

Description de l'image : Une personne se cogne la tête contre un mur à cause d'une chute, affichant des signes de commotion cérébrale dans un lieu public.

Question générée :
Quelles interventions un assistant IA spécialisé devrait-il recommander pour gérer une commotion cérébrale suite à une chute où une personne s'est cognée la tête contre un mur dans un lieu public ?

---

Description de l'image : Une victime d'une morsure de serpent venimeux immobilisée dans une zone rurale, avec des signes de choc et de douleur intense.

Question générée :
Quelles sont les actions prioritaires qu'un assistant IA spécialisé devrait-il recommander pour prendre en charge une morsure de serpent venimeux dans une zone rurale, en présence de signes de choc et de douleur intense ?

---

Description de l'image : Une personne victime d'un AVC, montrant des signes de paralysie d'un côté du corps dans un lieu de travail.

Question générée :
Quelles mesures de premiers secours un assistant IA spécialisé devrait-il suggérer pour gérer une personne victime d'un AVC présentant une paralysie d'un côté du corps dans un lieu de travail ?

---

Instructions pour Utiliser le Prompt :

1. Remplacement de la Description : Insérez la nouvelle description d'image à l'emplacement indiqué.
2. Génération de la Question : Laissez l'assistant IA générer la question pertinente basée sur les exemples fournis.
3. Vérification : Assurez-vous que la question générée est claire, pertinente et orientée vers l'action, conforme aux protocoles de premiers secours.'''
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
