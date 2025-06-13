# Classification_image_texte_et_deploiement
Ce projet de fin de module vise à construire un modèle de classification d’images multi-classes et un modèle de classification de texte en utilisant l’écosystème Hugging Face. Les modèles développés seront ensuite déployés via une API FastAPI conteneurisée avec Docker. Le projet sera géré de manière collaborative en binôme via Git

# Structure du projet 

classification_image_texte_et_deploiement/
│
├── README.md                          # Documentation principale du projet
├── requirements.txt                   # Dépendances Python
├── Dockerfile                         # Configuration Docker
├── docker-compose.yml                 # Configuration Docker Compose 
├── .gitignore                        # Fichiers à ignorer par Git
├── .dockerignore                     # Fichiers à ignorer par Docker
│
├── data/                             # Données du projet
│   ├── raw/                          # Données brutes
│   ├── processed/                    # Données prétraitées
│   └── samples/                      # Échantillons pour tests
│
├── models/                           # Modèles entraînés
│   ├── image_classification/         # Modèle de classification d'images
│   │
│   └── text_classification/          # Modèle de classification de texte
│
├── notebooks/                        # Notebooks Jupyter pour exploration
│
├── src/                              # Code source principal
│   ├── __init__.py
│   │
│   ├── data/                         # Scripts de gestion des données
│   │
│   ├── models/                       # Définition des modèles
│   │
│   ├── training/                     # Scripts d'entraînement
│   │
│   └── api/                          # Code de l'API FastAPI
│
├── tests/                            # Tests unitaires
│
├── scripts/                          # Scripts utilitaires 
│   ├──  # Téléchargement des datasets
│   ├── # Script d'entraînement global
│   ├── # Évaluation des modèles
│   └── # Script de déploiement
│
├── examples/                         # Exemples d'utilisation
│
├── docs/                             # Documentation détaillée
│   ├── installation.md               # Instructions d'installation
│   ├── api_documentation.md          # Documentation API
│   ├── model_architecture.md         # Architecture des modèles
│   └── deployment_guide.md           # Guide de déploiement
│
├── config/                           # Fichiers de configuration
│   ├──        # Config modèle images
│   ├──        # Config modèle texte
│   └──        # Config API
│
└── results/                          # Résultats et métriques
    ├── metrics/                      # Métriques d'évaluation
    ├── plots/                        # Graphiques et visualisations
    └── reports/                      # Rapports d'analyse