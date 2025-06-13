import os
from fastapi import FastAPI, HTTPException, File, UploadFile # Added File, UploadFile
from pydantic import BaseModel
import sys
import io # For image bytes
from PIL import Image # For image processing
import torch
import torchvision.transforms as T
import json

# Ajout du chemin src au PYTHONPATH pour permettre les imports relatifs
# depuis l'extérieur du dossier src/api, par exemple pour importer
# TextClassificationPipeline depuis src.training
# Cela est utile si l'API est lancée depuis la racine du projet.
# Si l'API est lancée depuis src/, alors src.training devrait être accessible.
# Pour plus de robustesse, nous ajoutons le chemin du parent de src.
# current_dir -> src/api
# parent_dir -> src
# grandparent_dir -> racine du projet
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

try:
    from training.text_classifier_hugging_face import TextClassificationPipeline
    from training.image_classifier import get_model # Import get_model
except ImportError as e:
    print(f"Erreur d'importation : {e}")
    print("Assurez-vous que le script text_classifier_hugging_face.py est accessible")
    print(f"PYTHONPATH actuel : {sys.path}")
    # Tenter un import relatif si le premier échoue (utile si lancé depuis src/api)
    try:
        from ..training.text_classifier_hugging_face import TextClassificationPipeline
        from ..training.image_classifier import get_model # Import get_model
    except ImportError:
        raise e # Relancer l'erreur originale si les deux échouent


app = FastAPI(title="API de Classification d'Image et de Texte", version="0.1.0")

# --- Configuration des chemins ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TEXT_MODEL_DIR_NAME = "text_classifier_hugging_face_model"
text_model_path = os.path.join(project_root, "models", TEXT_MODEL_DIR_NAME)

IMAGE_MODEL_DIR_NAME = "image_classifier_model"
image_model_dir_path = os.path.join(project_root, "models", IMAGE_MODEL_DIR_NAME)
IMAGE_MODEL_FILE = "image_classifier_best.pth"
IMAGE_CLASS_NAMES_FILE = "class_names.json"
image_model_file_path = os.path.join(image_model_dir_path, IMAGE_MODEL_FILE)
image_class_names_path = os.path.join(image_model_dir_path, IMAGE_CLASS_NAMES_FILE)

# --- Modèle de Classification de Texte ---
text_classifier = None

# --- Modèle de Classification d'Images ---
image_classifier = None
image_class_names = None
image_transform = None

class TextIn(BaseModel):
    text: str

class TextPredictionOut(BaseModel):
    text: str
    predictions: list # Liste de dictionnaires {'label': str, 'score': float}

class ImagePredictionOut(BaseModel):
    filename: str
    predicted_class: str
    confidence: float

@app.on_event("startup")
async def load_models():
    global text_classifier, image_classifier, image_class_names, image_transform
    print(f"Tentative de chargement du modèle de classification de texte depuis : {text_model_path}")
    if not os.path.exists(text_model_path):
        print(f"ERREUR : Le dossier du modèle de texte n'a pas été trouvé à l'emplacement : {text_model_path}")
    else:
        text_classifier = TextClassificationPipeline()
        try:
            text_classifier.load_model(text_model_path)
            print(f"Modèle de classification de texte chargé avec succès depuis {text_model_path}")
        except Exception as e:
            print(f"ERREUR lors du chargement du modèle de classification de texte : {e}")
            text_classifier = None

    print(f"Tentative de chargement du modèle de classification d'images depuis : {image_model_file_path}")
    if not os.path.exists(image_model_file_path):
        print(f"ERREUR : Le fichier du modèle d'image ('{IMAGE_MODEL_FILE}') n'a pas été trouvé à : {image_model_file_path}")
    elif not os.path.exists(image_class_names_path):
        print(f"ERREUR : Le fichier des noms de classes d'image ('{IMAGE_CLASS_NAMES_FILE}') n'a pas été trouvé à : {image_class_names_path}")
    else:
        try:
            with open(image_class_names_path, 'r') as f:
                image_class_names = json.load(f)
            print(f"Noms de classes d'image chargés depuis {image_class_names_path}")
            
            num_classes = len(image_class_names)
            image_classifier = get_model(num_classes=num_classes, pretrained=False) # pretrained=False car on charge nos propres poids
            
            # Déterminer le device
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            image_classifier.load_state_dict(torch.load(image_model_file_path, map_location=device))
            image_classifier.to(device)
            image_classifier.eval() # Mettre en mode évaluation
            print(f"Modèle de classification d'images chargé avec succès sur le device '{device}'.")

            # Définir les transformations pour l'inférence (similaires à val_transform)
            image_transform = T.Compose([
                T.Resize(256),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            print("Transformations d'image pour l'inférence définies.")

        except Exception as e:
            print(f"ERREUR lors du chargement du modèle de classification d'images : {e}")
            image_classifier = None
            image_class_names = None
            image_transform = None

@app.post("/predict/text", response_model=TextPredictionOut)
async def predict_text(text_input: TextIn):
    if text_classifier is None:
        raise HTTPException(status_code=503, detail="Modèle de classification de texte non chargé. Vérifiez les logs du serveur.")
    try:
        # La méthode predict de TextClassificationPipeline retourne une liste de dictionnaires,
        # même pour un seul texte. Nous prenons le premier (et unique) élément.
        prediction_result = text_classifier.predict([text_input.text])
        if not prediction_result:
            raise HTTPException(status_code=500, detail="La prédiction de texte a échoué ou retourné un résultat vide.")

        # prediction_result est une liste [{ "text": "...", "predictions": [{"label": ..., "score": ...}] }]
        # Nous voulons retourner directement le contenu de "predictions" pour le texte donné.
        return TextPredictionOut(
            text=prediction_result[0]["text"],
            predictions=prediction_result[0]["predictions"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la prédiction de texte : {str(e)}")

# --- Modèle de Classification d'Images ---
@app.post("/predict/image", response_model=ImagePredictionOut)
async def predict_image(file: UploadFile = File(...)):
    if image_classifier is None or image_class_names is None or image_transform is None:
        raise HTTPException(status_code=503, detail="Modèle de classification d'images non chargé. Vérifiez les logs du serveur.")

    try:
        contents = await file.read()
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Prétraitement de l'image
        img_tensor = image_transform(pil_image)
        img_tensor = img_tensor.unsqueeze(0) # Ajouter une dimension batch

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        img_tensor = img_tensor.to(device)

        with torch.no_grad():
            outputs = image_classifier(img_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)
        
        predicted_class_name = image_class_names[predicted_idx.item()]
        
        return ImagePredictionOut(
            filename=file.filename,
            predicted_class=predicted_class_name,
            confidence=confidence.item()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la prédiction d'image : {str(e)}")

@app.get("/")
async def root():
    return {"message": "Bienvenue sur l'API de Classification. Utilisez les endpoints /predict/text ou /predict/image."}

if __name__ == "__main__":
    import uvicorn
    # Note: Le lancement direct avec `python src/api/main.py` peut avoir des problèmes
    # avec les imports relatifs si PYTHONPATH n'est pas bien configuré.
    # Il est généralement préférable de lancer avec `uvicorn src.api.main:app --reload`
    # depuis la racine du projet.
    print("Pour lancer l'API, exécutez depuis la racine du projet :")
    print("uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000")
    # uvicorn.run(app, host="0.0.0.0", port=8000) # Décommentez pour un lancement direct simple (peut nécessiter des ajustements de PYTHONPATH)
