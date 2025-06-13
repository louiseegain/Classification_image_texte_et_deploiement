# Classification Image et Texte avec Déploiement API
**TP final cours de deeplearning**
**Réalisé par:**
 - **EGAIN Louise**
 - **LE D'HERVÉ Arthur**
 - **BRAI Yvan**


Ce projet vise à construire et déployer des modèles de classification d'images multi-classes et de classification de texte en utilisant l'écosystème Hugging Face. Les modèles sont ensuite exposés via une API FastAPI conteneurisée avec Docker.

## Fonctionnalités

*   **Classification d'images multi-classes :** Entraînement et évaluation d'un modèle capable de classifier des images dans plusieurs catégories.
*   **Classification de texte :** Entraînement et évaluation d'un modèle pour la classification de texte.
*   **API de déploiement :** Exposition des modèles entraînés via une API FastAPI.
*   **Conteneurisation :** Empaquetage de l'application et de ses dépendances dans un conteneur Docker pour un déploiement facile.
*   **Gestion de projet collaborative :** Utilisation de Git pour la gestion de versions et la collaboration.

## Technologies Utilisées

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)]()
[![Hugging Face Transformers](https://img.shields.io/badge/🤗%20Transformers-FFD000?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/docs/transformers/index)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Hugging Face Datasets](https://img.shields.io/badge/🤗%20Datasets-FFD000?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/docs/datasets/index)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Uvicorn](https://img.shields.io/badge/Uvicorn-009688?style=for-the-badge&logo=python&logoColor=white)](https://www.uvicorn.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/stable/)
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)

## Structure du Projet

```
.
├── README.md                   # Ce fichier
├── requirements.txt            # Dépendances Python du projet
├── config/                     # Fichiers de configuration
├── data/                       # Données brutes, intermédiaires et traitées
│   ├── external/
│   └── interim/
├── docs/                       # Documentation du projet
├── models/                     # Modèles sauvegardés et checkpoints
├── notebooks/                  # Jupyter notebooks pour l'exploration et l'expérimentation
├── results/                    # Résultats des entraînements et évaluations
│   ├── metrics/                # Métriques de performance
│   ├── plots/                  # Graphiques et visualisations
│   └── reports/                # Rapports générés
├── scripts/                    # Scripts utilitaires
├── src/                        # Code source du projet
│   ├── api/                    # Code de l'API FastAPI
│   ├── data/                   # Scripts pour le traitement des données
│   ├── models/                 # Définition des modèles
│   └── training/               # Scripts pour l'entraînement des modèles
│       └── text_classifier_hugging_face.py # Exemple de script d'entraînement
└── tests/                      # Tests unitaires et d'intégration
```

## Installation

1.  **Clonez le dépôt :**
    ```bash
    git clone <URL_DU_DEPOT>
    cd Classification_image_texte_et_deploiement
    ```

2.  **Créez un environnement virtuel et activez-le :** (Recommandé)
    ```bash
    python -m venv venv
    # Sur Windows
    .\venv\Scripts\activate
    # Sur macOS/Linux
    source venv/bin/activate
    ```

3.  **Installez les dépendances :**
    ```bash
    pip install -r requirements.txt
    ```

4.  **(Optionnel) Configuration de Docker :**
    Assurez-vous que Docker est installé et en cours d'exécution sur votre machine.

## Utilisation

### Entraînement des Modèles

Des scripts et notebooks pour l'entraînement des modèles se trouvent dans les répertoires `src/training/` et `notebooks/`.
Consultez la documentation spécifique dans ces répertoires pour des instructions détaillées.

Par exemple, pour entraîner le classifieur de texte :
```bash
python src/training/text_classifier_hugging_face.py
```
*(Adaptez cette commande en fonction de la configuration réelle de votre script)*

### Lancement de l'API FastAPI

1.  **Construire l'image Docker :** (Si un Dockerfile est présent à la racine ou dans `src/api/`)
    ```bash
    docker build -t image-texte-api .
    ```

2.  **Lancer le conteneur Docker :**
    ```bash
    docker run -d -p 8000:8000 image-texte-api
    ```
    L'API sera accessible à l'adresse `http://localhost:8000`.

*(Si vous n'utilisez pas Docker pour le développement local de l'API, vous pouvez lancer l'application FastAPI directement, par exemple avec Uvicorn. Assurez-vous que le point d'entrée de l'API est correctement configuré.)*

Exemple avec Uvicorn (si votre application FastAPI est dans `src/api/main.py` et l'instance FastAPI s'appelle `app`) :
```bash
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

### Accès à l'API

Une fois l'API lancée, vous pouvez interagir avec elle via des requêtes HTTP. La documentation interactive de l'API (généralement Swagger UI) est souvent disponible à l'adresse `http://localhost:8000/docs`.

Voici quelques exemples pour interroger les points de terminaison :

#### `/predict/text`

Ce point de terminaison attend une requête POST avec un corps JSON contenant une clé `"text"` et la chaîne de caractères à classifier.

**Exemple avec `curl` (Bash/Cmd) :**
```bash
curl -X POST "http://localhost:8000/predict/text" \
-H "Content-Type: application/json" \
-d '{"text": "Breaking news: Scientists discover new planet."}'
```

**Exemple avec PowerShell :**
```powershell
$headers = @{
    "Content-Type" = "application/json"
}
$body = '{"text": "Breaking news: Scientists discover new planet."}'
Invoke-RestMethod -Uri "http://localhost:8000/predict/text" -Method Post -Headers $headers -Body $body
```

#### `/predict/image`

Ce point de terminaison attend une requête POST avec des données de formulaire (`multipart/form-data`) contenant un fichier image sous la clé `"file"`.

**Exemple avec `curl` (Bash/Cmd) :**
Assurez-vous d'avoir une image nommée `votre_image.jpg` dans le répertoire où vous exécutez la commande.
```bash
curl -X POST "http://localhost:8000/predict/image" \
-F "file=@votre_image.jpg"
```

**Exemple avec PowerShell :**
Assurez-vous d'avoir une image (par exemple `votre_image.jpg`) et adaptez le chemin si nécessaire.
```powershell
$imagePath = "votre_image.jpg" # Remplacez par le chemin réel de votre image
$form = @{
    file = Get-Item -Path $imagePath
}
Invoke-RestMethod -Uri "http://localhost:8000/predict/image" -Method Post -Form $form
```
*(Note : Pour les images, assurez-vous que le format de l'image est compatible avec ce que le modèle attend, par exemple JPEG ou PNG.)*

### Importation de l'Image Docker

1.  **Importer l'image Docker depuis le fichier `.tar` :**
    Charger l'image dans son Docker local avec la commande suivante :
    ```bash
    docker load -i image-texte-api.tar
    ```
    Une fois cette commande exécutée, l'image `image-texte-api` sera disponible localement et pourra être lancée comme décrit dans la section "Lancement de l'API FastAPI".

## Performances des Modèles

### Modèle de Classification d'Images (ResNet18)

Entraîné sur le dataset "Image Classification - 32 Classes - Variety". Les métriques suivantes ont été obtenues sur l'ensemble de validation dédié (20% du dataset total) en utilisant le modèle `image_classifier_best.pth`:

- **Précision (Accuracy) :** 0.9951 (99.51%)
- **Précision (Precision Pondérée) :** 0.9952
- **Rappel (Recall Pondéré) :** 0.9951
- **Score F1 (Pondéré) :** 0.9951

Le modèle a été entraîné avec un arrêt anticipé (early stopping) basé sur la perte de validation (meilleure perte de validation observée pendant l'entraînement : ~0.0134), avec une patience de 5 époques. Le meilleur modèle (`image_classifier_best.pth`) a été sauvegardé à l'époque 11 de l'entraînement.

Un rapport de classification détaillé et une matrice de confusion générés par le script `src/training/evaluate_image_classifier.py` sont disponibles dans :
- `results/metrics/image_classifier_evaluation_metrics.json`
- `results/plots/image_classifier_confusion_matrix.png`

### Modèle de Classification de Texte (distilbert-base-uncased)

Entraîné sur le dataset AG News (classification de catégories d'actualités).

- **Perte d'Évaluation :** 0.1875
- **Précision d'Évaluation :** 0.9468 (94.68%)
- **Score F1 d'Évaluation :** 0.9469
- **Précision (Precision) d'Évaluation :** 0.9471
- **Rappel (Recall) d'Évaluation :** 0.9468

Ces métriques ont été obtenues après l'entraînement et l'évaluation du modèle sauvegardé dans `models/text_classifier_hugging_face_model`.

## Auteurs

- **EGAIN Louise** - *Développement et documentation* - [Profil GitHub](https://github.com/louiseegain)
- **LE D'HERVÉ Arthur** - *Développement et documentation* - [Profil GitHub](https://github.com/Arthur-LDH)
- **BRAI Yvan** - *Développement et documentation* - [Profil GitHub](https://github.com/zehelh)
