# Utiliser une image Python officielle comme image de base
FROM python:3.12-slim

# Définir le répertoire de travail dans le conteneur
WORKDIR /app

# Copier le fichier des dépendances en premier pour profiter de la mise en cache de Docker
COPY requirements.txt .

# Installer les dépendances
# --no-cache-dir réduit la taille de l'image
# --upgrade pip s'assure que pip est à jour
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copier le code source de l'application (y compris l'API et le code d'entraînement si nécessaire pour l'API)
COPY ./src /app/src

# Copier les modèles entraînés dans le conteneur
# Assurez-vous que le chemin vers les modèles dans votre API (main.py) est cohérent avec ce chemin
COPY ./models /app/models

# Exposer le port sur lequel l'API FastAPI s'exécutera
EXPOSE 8000

# Commande pour lancer l'application FastAPI avec Uvicorn
# Cela suppose que votre instance FastAPI s'appelle 'app' dans 'src/api/main.py'
# Et que Uvicorn est listé dans requirements.txt
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
