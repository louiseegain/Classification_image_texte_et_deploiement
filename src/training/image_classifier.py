import os
from pathlib import Path
import collections
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from PIL import Image # For handling images
import time # To time training
import json # To save class names

# Define project root assuming this script is in src/training/
PROJECT_ROOT = Path(__file__).resolve().parents[2]
# The user confirmed the dataset is in: data/external/Image_Classification_Dataset_32_Classes/
DATA_DIR_RELATIVE = "data/external/Image_Classification_Dataset_32_Classes"
REPORTS_DIR_RELATIVE = "results/reports"
PLOTS_DIR_RELATIVE = "results/plots/image_samples" # Specific path for image samples
MODEL_SAVE_DIR_RELATIVE = "models/image_classifier_model" # Path to save the trained image model

DATA_PATH = PROJECT_ROOT / DATA_DIR_RELATIVE
REPORTS_PATH = PROJECT_ROOT / REPORTS_DIR_RELATIVE
PLOTS_PATH = PROJECT_ROOT / PLOTS_DIR_RELATIVE
MODEL_SAVE_PATH = PROJECT_ROOT / MODEL_SAVE_DIR_RELATIVE

def load_image_dataset(data_dir, transform=None):
    """
    Loads an image dataset using ImageFolder.
    data_dir should be the path to the directory containing class subfolders.
    """
    if not data_dir.exists():
        print(f"Erreur : Le répertoire de données est introuvable à {data_dir}")
        print("Veuillez vous assurer que le dataset 'Image_Classification_Dataset_32_Classes' est téléchargé et extrait dans 'data/external/'.")
        return None
    
    # Check if data_dir contains subdirectories (classes)
    if not any(item.is_dir() for item in data_dir.iterdir()):
        print(f"Erreur : Aucun sous-répertoire de classe trouvé dans {data_dir}.")
        print("Le répertoire spécifié doit contenir des sous-dossiers, un pour chaque classe d'images.")
        print(f"Contenu actuel de {data_dir}: {[item.name for item in data_dir.iterdir()]}")
        return None

    if transform is None:
        transform = transforms.Compose([
            transforms.Resize((224, 224)), # Standard size for many pre-trained models
            transforms.ToTensor()
        ])
    
    try:
        dataset = datasets.ImageFolder(root=str(data_dir), transform=transform)
        print(f"Dataset chargé depuis {data_dir}")
        if len(dataset) == 0:
            print(f"Attention : Le dataset chargé depuis {data_dir} est vide.")
            print("Vérifiez que les images sont correctement placées dans les sous-dossiers de classe.")
        else:
            print(f"{len(dataset)} images trouvées dans {len(dataset.classes)} classes.")
        return dataset
    except Exception as e:
        print(f"Erreur lors du chargement du dataset avec ImageFolder depuis {data_dir}: {e}")
        return None

def explore_dataset_stats(dataset, report_file_path):
    """
    Generates a report with dataset statistics.
    """
    if dataset is None or len(dataset) == 0:
        print("Dataset non chargé ou vide. Skipping statistics generation.")
        return

    report_file_path.parent.mkdir(parents=True, exist_ok=True)
    
    num_images = len(dataset)
    class_names = dataset.classes
    num_classes = len(class_names)
    
    class_counts = collections.Counter(dataset.targets)
    class_counts_named = {class_names[i]: count for i, count in class_counts.items()}

    sample_images_info = []
    max_samples_for_info = min(5, num_images) # Ensure we don't try to sample more images than available
    # dataset.samples is a list of (image_path, class_index)
    # We use this to get original image paths for stats
    for i in range(max_samples_for_info):
        img_path, _ = dataset.samples[i] 
        try:
            with Image.open(img_path) as pil_img:
                sample_images_info.append({
                    "filename": os.path.basename(img_path),
                    "size": pil_img.size,
                    "mode": pil_img.mode
                })
        except Exception as e:
            sample_images_info.append({
                "filename": os.path.basename(img_path),
                "error": str(e)
            })

    with open(report_file_path, 'w', encoding='utf-8') as f:
        f.write("Rapport d'Exploration du Dataset d'Images\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Chemin du Dataset : {dataset.root}\n")
        f.write(f"Nombre total d'images : {num_images}\n")
        f.write(f"Nombre de classes : {num_classes}\n\n")
        f.write("Classes et nombre d'images par classe :\n")
        for class_name, count in sorted(class_counts_named.items()):
            f.write(f"- {class_name}: {count}\n")
        f.write("\n")
        f.write(f"Informations sur des exemples d'images (les {len(sample_images_info)} premières images) :\n")
        for info in sample_images_info:
            if "error" in info:
                f.write(f"- {info['filename']}: Erreur de lecture ({info['error']})\n")
            else:
                f.write(f"- {info['filename']}: Taille={info['size']}, Mode={info['mode']}\n")
        
    print(f"Rapport des statistiques du dataset sauvegardé : {report_file_path}")

def save_sample_images(dataset, plots_dir, num_samples_per_class=3):
    """
    Saves a few sample images from each class from their original paths.
    """
    if dataset is None or len(dataset) == 0:
        print("Dataset non chargé ou vide. Skipping sample image saving.")
        return

    plots_dir.mkdir(parents=True, exist_ok=True)
    
    class_names = dataset.classes
    # dataset.samples is a list of (filepath, class_index)
    # Group image paths by class_idx
    class_images_paths = {class_idx: [] for class_idx in range(len(class_names))}
    for img_path, class_idx in dataset.samples:
        class_images_paths[class_idx].append(img_path)

    for class_idx, class_name in enumerate(class_names):
        class_specific_plot_dir = plots_dir / class_name
        class_specific_plot_dir.mkdir(parents=True, exist_ok=True)
        
        image_paths_for_class = class_images_paths[class_idx]
        
        if not image_paths_for_class:
            print(f"Aucune image trouvée pour la classe : {class_name}")
            continue

        print(f"Sauvegarde des échantillons pour la classe : {class_name}...")
        saved_count = 0
        for i, img_path_str in enumerate(image_paths_for_class):
            if saved_count >= num_samples_per_class:
                break
            try:
                img_path = Path(img_path_str) # Ensure it's a Path object
                img = Image.open(img_path)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                sample_filename = f"{class_name}_sample_{saved_count+1}{img_path.suffix}"
                img.save(class_specific_plot_dir / sample_filename)
                saved_count += 1
            except Exception as e:
                print(f"Erreur lors de la sauvegarde de l'image échantillon {img_path_str} pour la classe {class_name}: {e}")
        
        print(f"{saved_count} échantillons sauvegardés pour la classe {class_name} dans {class_specific_plot_dir}")

def get_model(num_classes, pretrained=True):
    """
    Charge un modèle ResNet18 pré-entraîné et adapte sa dernière couche.

    Justification du choix (ResNet18) :
    ResNet18 est choisi comme point de départ car il offre un bon compromis 
    entre performance et complexité computationnelle. En tant que modèle 
    pré-entraîné sur ImageNet, il a déjà appris des caractéristiques visuelles 
    générales qui sont souvent transférables à d'autres tâches de classification 
    d'images. Cela permet un entraînement plus rapide et potentiellement de 
    meilleures performances, surtout avec des datasets de taille modérée. 
    Sa relative légèreté par rapport à des modèles plus profonds (comme ResNet50+) 
    le rend aussi plus rapide à entraîner et moins gourmand en ressources.
    """
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    
    # Geler les poids des couches pré-entraînés si l'on fait du fine-tuning
    # for param in model.parameters():
    #     param.requires_grad = False # Décommenter pour geler les couches initiales
        
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes) # Remplacer la dernière couche pour nos N classes
    return model

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    """
    Entraîne le modèle et l'évalue à chaque époque.
    """
    model.to(device)
    best_val_accuracy = 0.0
    best_val_loss = float('inf') # Pour l'arrêt anticipé basé sur la perte
    epochs_no_improve = 0
    
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # Phase d'entraînement
        model.train()
        running_loss = 0.0
        running_corrects = 0
        
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)
        
        # Phase de validation
        val_loss, val_acc = evaluate_model(model, val_loader, criterion, device)
        
        end_time = time.time()
        epoch_duration = end_time - start_time
        
        print(f"Époque {epoch+1}/{num_epochs} - Durée: {epoch_duration:.2f}s")
        print(f"  Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
        
        # Sauvegarder le meilleur modèle (basé sur la précision de validation ET la perte pour l'arrêt anticipé)
        if val_loss < best_val_loss: # Changement ici pour se baser sur la perte pour l'arrêt anticipé
            best_val_loss = val_loss
            best_val_accuracy = val_acc # Mettre à jour aussi la meilleure précision
            save_model(model, MODEL_SAVE_PATH, "image_classifier_best.pth")
            print(f"  Meilleur modèle (basé sur Val Loss) sauvegardé avec Val Loss: {best_val_loss:.4f}, Val Acc: {best_val_accuracy:.4f}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"  Aucune amélioration de Val Loss pendant {epochs_no_improve} époque(s).")

        # Critère d'arrêt anticipé
        # Utiliser la patience définie dans la configuration principale
        if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
            print(f"\\\\nArrêt anticipé ! Aucune amélioration de Val Loss pendant {EARLY_STOPPING_PATIENCE} époques consécutives.")
            break
            
    print(f"Entraînement terminé. Meilleure Val Loss: {best_val_loss:.4f}, Meilleure Val Acc: {best_val_accuracy:.4f}")
    return model

def evaluate_model(model, data_loader, criterion, device):
    """
    Évalue le modèle sur un ensemble de données donné.
    """
    model.eval() # Mettre le modèle en mode évaluation
    running_loss = 0.0
    running_corrects = 0
    
    with torch.no_grad(): # Pas besoin de calculer les gradients pendant l'évaluation
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            
    total_loss = running_loss / len(data_loader.dataset)
    total_acc = running_corrects.double() / len(data_loader.dataset)
    
    return total_loss, total_acc

def save_model(model, save_dir, model_filename="image_classifier_final.pth"):
    """
    Sauvegarde l'état du modèle.
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    model_path = save_dir / model_filename
    torch.save(model.state_dict(), model_path)
    print(f"Modèle sauvegardé : {model_path}")

if __name__ == "__main__":
    # --- Configuration ---
    NUM_EPOCHS = 20 # Nombre d'époques
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    TRAIN_VAL_SPLIT_RATIO = 0.8 # 80% pour l'entraînement, 20% pour la validation
    EARLY_STOPPING_PATIENCE = 5 # Nombre d'époques à attendre avant l'arrêt anticipé
    # Utiliser un nombre plus petit d'échantillons par classe pour l'exploration si nécessaire
    # NUM_SAMPLES_PER_CLASS_EXPLORATION = 1 
    NUM_SAMPLES_PER_CLASS_EXPLORATION = 5


    print(f"Racine du projet : {PROJECT_ROOT}")
    print(f"Chemin des données : {DATA_PATH}")
    print(f"Chemin des rapports : {REPORTS_PATH}")
    print(f"Chemin des graphiques : {PLOTS_PATH}")
    print(f"Chemin de sauvegarde du modèle : {MODEL_SAVE_PATH}")

    REPORTS_PATH.mkdir(parents=True, exist_ok=True)
    PLOTS_PATH.mkdir(parents=True, exist_ok=True)
    MODEL_SAVE_PATH.mkdir(parents=True, exist_ok=True)


    # --- Transformations des données ---
    # Transformations plus robustes pour l'entraînement
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # Transformations pour la validation (pas d'augmentation, juste normalisation)
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print("\\nChargement du dataset d'images complet pour l'entraînement...")
    # Charger le dataset avec les transformations de validation initialement pour l'exploration
    # puis nous allons le diviser et appliquer les transformations spécifiques train/val
    full_dataset = load_image_dataset(DATA_PATH, transform=val_transform) # Utiliser val_transform pour le dataset complet initial

    if full_dataset and len(full_dataset) > 0:
        num_classes = len(full_dataset.classes)
        print(f"Nombre de classes détectées : {num_classes}")

        # Save class names
        class_names_path = MODEL_SAVE_PATH / "class_names.json"
        with open(class_names_path, 'w') as f:
            json.dump(full_dataset.classes, f)
        print(f"Noms de classes sauvegardés dans {class_names_path}")

        # --- 1. Exploration des données (si pas déjà fait ou pour confirmer) ---
        # print("\\nExploration des statistiques du dataset...")
        # report_file = REPORTS_PATH / "image_dataset_exploration_report.txt"
        # explore_dataset_stats(full_dataset, report_file) # full_dataset ici utilise val_transform

        # print("\\nSauvegarde des images échantillons...")
        # save_sample_images(full_dataset, PLOTS_PATH, num_samples_per_class=NUM_SAMPLES_PER_CLASS_EXPLORATION)
        # print("\\nExploration terminée.")
        
        # --- 2. Préparation des données pour l'entraînement ---
        print("\\nPréparation des données pour l'entraînement et la validation...")
        train_size = int(TRAIN_VAL_SPLIT_RATIO * len(full_dataset))
        val_size = len(full_dataset) - train_size
        
        # Note: random_split crée des Subsets. Pour appliquer des transformations différentes,
        # il est souvent plus simple de recréer les datasets ImageFolder pour train/val
        # ou d'encapsuler les Subsets dans une classe Dataset personnalisée qui applique les transformations.
        # Pour simplifier ici, nous allons utiliser les mêmes transformations pour les deux après le split,
        # ou mieux, nous allons créer des datasets distincts si possible, ou appliquer les transformations
        # au niveau des DataLoaders si les datasets sont des Subsets.
        # Une approche plus propre est de créer des datasets distincts pour train et val avec leurs propres transforms.
        # Cependant, ImageFolder charge tout un dossier.
        # Une solution commune est de diviser les *indices* puis de créer des `Subset`s,
        # et d'avoir une transformation conditionnelle ou un wrapper de Dataset.

        # Pour cet exemple, nous allons créer des Subsets et leur assigner les transformations
        # en les encapsulant ou en s'assurant que les DataLoaders les gèrent.
        # La manière la plus simple avec `random_split` est que les transformations sont attachées au `full_dataset`
        # et donc partagées. Pour des transformations distinctes (train_transform, val_transform),
        # il faut une approche plus avancée (créer des dossiers train/val séparés ou wrapper de Dataset).

        # Compromis pour ce script : nous allons utiliser `random_split` et les transformations
        # définies ci-dessus. Le `full_dataset` a été chargé avec `val_transform`.
        # Pour appliquer `train_transform` au subset d'entraînement, nous devons le réassigner.
        
        train_subset, val_subset = random_split(full_dataset, [train_size, val_size])

        # Pour appliquer des transformations différentes aux subsets:
        # Créez des classes wrapper de Dataset ou modifiez les attributs .dataset.transform
        # Ceci est un peu un hack mais fonctionne pour ImageFolder subsets:
        # Il faut accéder à l'objet dataset sous-jacent du Subset pour changer sa transformation.
        # Cependant, `full_dataset` est partagé.
        # La meilleure pratique est d'avoir des dossiers train/val séparés sur le disque.
        # Si ce n'est pas possible, on peut créer des instances de `DatasetFolder` distinctes
        # en filtrant les fichiers, ou utiliser une bibliothèque qui gère mieux cela.

        # Pour garder simple ici, nous allons créer les DataLoaders et les transformations
        # seront celles du `full_dataset` (val_transform).
        # Pour une VRAIE augmentation sur train, il faudrait une structure de données différente.
        # Alternative: créer deux instances de ImageFolder si on a des listes de fichiers pour train/val.

        print("Chargement du dataset pour l'entraînement (avec augmentation)...")
        train_dataset_for_split = load_image_dataset(DATA_PATH, transform=train_transform)
        print("Chargement du dataset pour la validation (sans augmentation)...")
        val_dataset_for_split = load_image_dataset(DATA_PATH, transform=val_transform)

        if train_dataset_for_split and val_dataset_for_split:
            # Assurer la reproductibilité du split
            torch.manual_seed(42)
            train_indices, val_indices = random_split(range(len(full_dataset)), [train_size, val_size])

            train_dataset = torch.utils.data.Subset(train_dataset_for_split, train_indices.indices)
            val_dataset = torch.utils.data.Subset(val_dataset_for_split, val_indices.indices)

            print(f"Taille de l'ensemble d'entraînement : {len(train_dataset)} images")
            print(f"Taille de l'ensemble de validation : {len(val_dataset)} images")

            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
            val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
            
            print("DataLoaders créés.")

            # --- 3. Définition et Entraînement du Modèle ---
            print("\\nDéfinition du modèle...")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Utilisation du device : {device}")

            model = get_model(num_classes=num_classes, pretrained=True)
            model.to(device)

            criterion = nn.CrossEntropyLoss()
            # Entraîner seulement la dernière couche (classifier)
            # optimizer = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE) 
            # Entraîner toutes les couches (fine-tuning plus profond)
            optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


            print("\\nDébut de l'entraînement du modèle...")
            trained_model = train_model(model, train_loader, val_loader, criterion, optimizer, NUM_EPOCHS, device)
            
            # --- 4. Sauvegarde du modèle final ---
            print("\\nSauvegarde du modèle final entraîné...")
            save_model(trained_model, MODEL_SAVE_PATH, "image_classifier_final_after_epochs.pth")

            print("\\nScript d'entraînement du modèle image terminé.")
        else:
            print("Erreur lors du chargement des datasets pour l'entraînement/validation.")

    else:
        print("\\nLe script d'entraînement n'a pas pu s'exécuter (problème de chargement du dataset initial ou dataset vide).")
        if DATA_PATH.exists():
            print(f"Vérifiez la structure du contenu de : {DATA_PATH}")
            print("Il devrait contenir des sous-dossiers pour chaque classe, remplis d'images.")
        else:
            print(f"Le chemin {DATA_PATH} n'existe pas.")
