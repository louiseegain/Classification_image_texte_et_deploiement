"""
Script d'exploration du dataset d'images
Auteur: [Votre Nom]
Date: 13 Juin 2025
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from collections import Counter
import random
from PIL import Image

def explore_dataset_structure():
    """Explorer la structure de base du dataset"""
    print("EXPLORATION DU DATASET D'IMAGES")
    print("=" * 60)
    
    # Chemin vers les données
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / "data" / "raw"
    
    print(f"Dossier de données : {data_dir}")
    
    if not data_dir.exists():
        print("[ERROR] Le dossier data/raw n'existe pas")
        return None
    
    # Lister toutes les classes (dossiers)
    all_items = [item for item in data_dir.iterdir() if item.is_dir()]
    classes = [item.name for item in all_items if item.name != "image"]  # Exclure le dossier vide
    
    print(f"\n[DEBUG] Tous les dossiers trouvés : {[item.name for item in all_items]}")
    print(f"[DEBUG] Classes filtrées : {classes}")
    
    if not classes:
        print("[ERROR] Aucune classe détectée!")
        print("[INFO] Vérifiez que le dataset est bien téléchargé dans data/raw/")
        return None
    
    print(f"\n[INFO] Nombre de classes détectées : {len(classes)}")
    print("\n[CLASSES] Classes disponibles :")
    for i, class_name in enumerate(sorted(classes), 1):
        print(f"  {i:2d}. {class_name}")
    
    return data_dir, classes

def analyze_class_distribution(data_dir, classes):
    """Analyser la distribution des images par classe"""
    print(f"\n[ANALYSIS] Analyse de la distribution par classe")
    print("-" * 50)
    print(f"{'Classe':<15} {'Nb Images':<10} {'Taille Moy':<12}")
    print("-" * 40)
    
    class_data = []
    total_images = 0
    
    for class_name in sorted(classes):
        class_path = data_dir / class_name
        
        # Compter les images (fichiers .png, .jpg, etc.)
        image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif', '.webp'}
        images = [f for f in class_path.iterdir() 
                 if f.is_file() and f.suffix.lower() in image_extensions]
        
        count = len(images)
        total_images += count
        
        # Calculer la taille moyenne (sur les 5 premiers fichiers)
        if images:
            sample_sizes = []
            for img in images[:5]:
                try:
                    size_mb = img.stat().st_size / (1024 * 1024)
                    sample_sizes.append(size_mb)
                except:
                    pass
            avg_size = np.mean(sample_sizes) if sample_sizes else 0
        else:
            avg_size = 0
        
        class_data.append({
            'classe': class_name,
            'count': count,
            'avg_size_mb': avg_size
        })
        
        print(f"{class_name:<15} {count:<10} {avg_size:.2f} MB")
    
    print(f"\n[SUMMARY] Total images : {total_images:,}")
    if len(classes) > 0:
        print(f"[SUMMARY] Moyenne par classe : {total_images // len(classes)}")
    else:
        print(f"[ERROR] Aucune classe détectée!")
    
    return class_data, total_images

def analyze_imbalance(class_data):
    """Analyser le déséquilibre du dataset"""
    print(f"\n[BALANCE] Analyse du déséquilibre")
    print("-" * 30)
    
    # Créer un DataFrame pour l'analyse
    df = pd.DataFrame(class_data)
    df = df.sort_values('count', ascending=False)
    
    print(f"Top 5 classes avec le plus d'images :")
    for _, row in df.head().iterrows():
        print(f"  {row['classe']:<15} : {row['count']} images")
    
    print(f"\nTop 5 classes avec le moins d'images :")
    for _, row in df.tail().iterrows():
        print(f"  {row['classe']:<15} : {row['count']} images")
    
    # Analyse du déséquilibre
    min_count = df['count'].min()
    max_count = df['count'].max()
    ratio = max_count / min_count if min_count > 0 else 0
    
    print(f"\n[BALANCE] Statistiques de déséquilibre :")
    print(f"  Min images : {min_count}")
    print(f"  Max images : {max_count}")
    print(f"  Ratio max/min : {ratio:.2f}")
    print(f"  Écart-type : {df['count'].std():.1f}")
    
    if ratio > 3:
        print("  [STATUS] Dataset fortement déséquilibré")
        print("  [RECOMMENDATION] Utiliser des techniques de rééquilibrage")
    elif ratio > 2:
        print("  [STATUS] Dataset modérément déséquilibré")
        print("  [RECOMMENDATION] Augmentation de données recommandée")
    else:
        print("  [STATUS] Dataset équilibré")
    
    return df

def sample_images_analysis(data_dir, classes, sample_size=50):
    """Analyser un échantillon d'images pour les propriétés"""
    print(f"\n[IMAGES] Analyse d'un échantillon d'images")
    print("-" * 40)
    
    # Échantillonner des images
    sampled_images = []
    for class_name in classes[:8]:  # Limiter à 8 classes pour l'échantillon
        class_path = data_dir / class_name
        image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
        images = [f for f in class_path.iterdir() 
                 if f.is_file() and f.suffix.lower() in image_extensions]
        
        if images:
            sample_count = min(sample_size // len(classes), len(images))
            selected = random.sample(images, sample_count)
            sampled_images.extend(selected)
    
    print(f"Analyse de {len(sampled_images)} images échantillonnées...")
    
    # Analyser les propriétés
    widths, heights, sizes_mb, formats = [], [], [], []
    
    for img_path in sampled_images:
        try:
            with Image.open(img_path) as img:
                width, height = img.size
                widths.append(width)
                heights.append(height)
                formats.append(img.format)
                
                # Taille du fichier
                size_mb = img_path.stat().st_size / (1024 * 1024)
                sizes_mb.append(size_mb)
                
        except Exception as e:
            continue
    
    if widths:
        print(f"\n[DIMENSIONS] Statistiques des dimensions :")
        print(f"  Largeurs  - Min: {min(widths):4d}, Max: {max(widths):4d}, Moyenne: {np.mean(widths):.1f}")
        print(f"  Hauteurs  - Min: {min(heights):4d}, Max: {max(heights):4d}, Moyenne: {np.mean(heights):.1f}")
        print(f"  Ratios    - Min: {min(np.array(widths)/np.array(heights)):.2f}, Max: {max(np.array(widths)/np.array(heights)):.2f}")
        
        print(f"\n[FILES] Statistiques des fichiers :")
        print(f"  Taille    - Min: {min(sizes_mb):.2f}MB, Max: {max(sizes_mb):.2f}MB, Moyenne: {np.mean(sizes_mb):.2f}MB")
        
        print(f"\n[FORMATS] Formats détectés :")
        format_counts = Counter(formats)
        for fmt, count in format_counts.most_common():
            percentage = (count / len(formats)) * 100
            print(f"  {fmt:<8} : {count:3d} images ({percentage:.1f}%)")
        
        # Recommandations de preprocessing
        print(f"\n[PREPROCESSING] Recommandations :")
        
        # Taille cible
        target_size = 224  # Standard pour les modèles pré-entraînés
        print(f"  1. Redimensionner toutes les images à {target_size}x{target_size} pixels")
        
        # Normalisation
        print(f"  2. Normaliser avec les valeurs ImageNet (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])")
        
        # Augmentation
        if max(np.array(widths)/np.array(heights)) > 2 or min(np.array(widths)/np.array(heights)) < 0.5:
            print(f"  3. Utiliser des transformations géométriques (rotation, flip, crop)")
        
        print(f"  4. Appliquer de l'augmentation de données pour équilibrer les classes")

def generate_metadata(data_dir, classes):
    """Générer des métadonnées pour l'entraînement"""
    print(f"\n[METADATA] Génération des métadonnées")
    print("-" * 35)
    
    # Créer un DataFrame avec toutes les images
    all_images_data = []
    
    for class_idx, class_name in enumerate(sorted(classes)):
        class_path = data_dir / class_name
        image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif', '.webp'}
        
        for img_path in class_path.iterdir():
            if img_path.is_file() and img_path.suffix.lower() in image_extensions:
                all_images_data.append({
                    'image_path': str(img_path),
                    'class_name': class_name,
                    'class_idx': class_idx,
                    'filename': img_path.name,
                    'relative_path': str(img_path.relative_to(data_dir))
                })
    
    df_metadata = pd.DataFrame(all_images_data)
    
    print(f"Métadonnées créées pour {len(df_metadata)} images")
    print(f"Classes indexées de 0 à {df_metadata['class_idx'].max()}")
    
    # Sauvegarder les métadonnées
    processed_dir = data_dir.parent / "processed"
    processed_dir.mkdir(exist_ok=True)
    
    metadata_path = processed_dir / "image_metadata.csv"
    df_metadata.to_csv(metadata_path, index=False)
    
    # Sauvegarder la liste des classes
    classes_path = processed_dir / "classes.txt"
    with open(classes_path, 'w') as f:
        for class_name in sorted(classes):
            f.write(f"{class_name}\n")
    
    print(f"[SAVED] Métadonnées sauvegardées : {metadata_path}")
    print(f"[SAVED] Liste des classes : {classes_path}")
    
    return df_metadata

def main():
    """Fonction principale d'exploration"""
    # Seed pour reproductibilité
    random.seed(42)
    np.random.seed(42)
    
    # 1. Explorer la structure
    result = explore_dataset_structure()
    if result is None:
        return
    
    data_dir, classes = result
    
    # 2. Analyser la distribution
    class_data, total_images = analyze_class_distribution(data_dir, classes)
    
    # 3. Analyser le déséquilibre
    df_balance = analyze_imbalance(class_data)
    
    # 4. Analyser des échantillons d'images
    sample_images_analysis(data_dir, classes)
    
    # 5. Générer les métadonnées
    df_metadata = generate_metadata(data_dir, classes)
    
    # Résumé final
    print(f"\n" + "=" * 60)
    print(f"[SUCCESS] EXPLORATION TERMINÉE")
    print(f"=" * 60)
    print(f"Dataset analysé : {len(classes)} classes, {total_images:,} images")
    print(f"Métadonnées générées et sauvegardées")
    print(f"")
    print(f"[NEXT] Prochaines étapes :")
    print(f"  1. Créer le modèle CNN (ResNet/EfficientNet)")
    print(f"  2. Implémenter l'entraînement")
    print(f"  3. Évaluer les performances")
    print(f"  4. Sauvegarder le modèle")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[STOP] Exploration interrompue par l'utilisateur")
    except Exception as e:
        print(f"\n[ERROR] Erreur lors de l'exploration : {e}")
        import traceback
        traceback.print_exc()