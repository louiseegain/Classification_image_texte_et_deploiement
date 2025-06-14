
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd
from pathlib import Path
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Ajouter le chemin src
sys.path.append(str(Path(__file__).parent.parent / "src"))

from models.image_classifier import create_model_from_config, count_parameters

class ImageDataset(Dataset):
    """Dataset personnalisé pour les images"""
    
    def __init__(self, dataframe, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        # Charger l'image
        img_path = self.df.iloc[idx]['image_path']
        image = Image.open(img_path).convert('RGB')
        
        # Appliquer les transformations
        if self.transform:
            image = self.transform(image)
        
        # Label
        label = self.df.iloc[idx]['class_idx']
        
        return image, label

def create_transforms():
    """Créer les transformations pour l'entraînement et la validation"""
    
    # Transformations pour l'entraînement (avec augmentation)
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Transformations pour la validation (sans augmentation)
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def prepare_datasets():
    """Préparer les datasets d'entraînement et de validation"""
    print("[DATA] Préparation des datasets...")
    
    # Charger les métadonnées
    base_dir = Path(__file__).parent.parent
    metadata_path = base_dir / "data" / "processed" / "image_metadata.csv"
    
    if not metadata_path.exists():
        print(f"[ERROR] Métadonnées non trouvées : {metadata_path}")
        print("[INFO] Exécutez d'abord : python scripts/explore_dataset.py")
        return None
    
    df = pd.read_csv(metadata_path)
    print(f"[INFO] {len(df)} images chargées")
    
    # Division train/val/test
    train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df['class_name'], random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['class_name'], random_state=42)
    
    print(f"[SPLIT] Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Créer les transformations
    train_transform, val_transform = create_transforms()
    
    # Créer les datasets
    train_dataset = ImageDataset(train_df, transform=train_transform)
    val_dataset = ImageDataset(val_df, transform=val_transform)
    test_dataset = ImageDataset(test_df, transform=val_transform)
    
    return train_dataset, val_dataset, test_dataset

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Entraîner le modèle pour une époque"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (images, labels) in enumerate(dataloader):
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Statistiques
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Progress
        if batch_idx % 20 == 0:
            print(f"\r[TRAIN] Batch {batch_idx}/{len(dataloader)} | "
                  f"Loss: {loss.item():.4f} | "
                  f"Acc: {100.*correct/total:.2f}%", end="")
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc

def validate_epoch(model, dataloader, criterion, device):
    """Valider le modèle pour une époque"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc

def train_model():
    """Fonction principale d'entraînement"""
    print("ENTRAÎNEMENT DU MODÈLE DE CLASSIFICATION D'IMAGES")
    print("=" * 60)
    
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[DEVICE] Utilisation de : {device}")
    
    # Paramètres d'entraînement
    batch_size = 32
    num_epochs = 10  # Commencer petit pour tester
    learning_rate = 0.001
    
    # Préparer les données
    datasets = prepare_datasets()
    if datasets is None:
        return
    
    train_dataset, val_dataset, test_dataset = datasets
    
    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    print(f"[LOADERS] Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Créer le modèle
    config_path = Path(__file__).parent.parent / "config" / "image_model_config.yaml"
    if config_path.exists():
        model, config = create_model_from_config(config_path)
    else:
        model, config = create_model_from_config()
        print("[WARNING] Configuration par défaut utilisée")
    
    model = model.to(device)
    
    # Afficher les paramètres du modèle
    params = count_parameters(model)
    print(f"[MODEL] Architecture: {config['model']['architecture']}")
    print(f"[MODEL] Paramètres entraînables: {params['trainable']:,}")
    
    # Loss et optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    # Variables pour le suivi
    best_val_acc = 0.0
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    
    print(f"\n[START] Début de l'entraînement...")
    start_time = time.time()
    
    for epoch in range(num_epochs):
        print(f"\n[EPOCH {epoch+1}/{num_epochs}]")
        
        # Entraînement
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        print(f"\n[TRAIN] Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
        
        # Validation
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
        print(f"[VAL] Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
        
        # Scheduler
        scheduler.step()
        
        # Sauvegarder les métriques
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Sauvegarder le meilleur modèle
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            
            # Créer le dossier de sauvegarde
            save_dir = Path(__file__).parent.parent / "models" / "image_classification"
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # Sauvegarder le modèle
            model_path = save_dir / "best_model.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'config': config
            }, model_path)
            
            print(f"[SAVE] Nouveau meilleur modèle sauvegardé: {val_acc:.2f}%")
    
    training_time = time.time() - start_time
    
    print(f"\n" + "=" * 60)
    print(f"[SUCCESS] ENTRAÎNEMENT TERMINÉ")
    print(f"[TIME] Durée totale: {training_time/60:.1f} minutes")
    print(f"[BEST] Meilleure validation accuracy: {best_val_acc:.2f}%")
    print(f"[SAVED] Modèle sauvegardé dans: models/image_classification/")
    
    return model, (train_losses, train_accs, val_losses, val_accs)

if __name__ == "__main__":
    try:
        train_model()
    except KeyboardInterrupt:
        print("\n[STOP] Entraînement interrompu par l'utilisateur")
    except Exception as e:
        print(f"\n[ERROR] Erreur pendant l'entraînement: {e}")
        import traceback
        traceback.print_exc()