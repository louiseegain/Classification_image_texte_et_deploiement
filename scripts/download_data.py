"""
Script de téléchargement avec monitoring avancé
"""
import time
import threading
import os
from pathlib import Path

stop_animation = False

def monitor_download_folder(raw_dir):
    """Surveiller la taille du dossier pendant le téléchargement"""
    chars = ['|', '/', '-', '\\']
    i = 0
    start_time = time.time()
    
    while not stop_animation:
        elapsed = time.time() - start_time
        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)
        time_str = f"{minutes:02d}:{seconds:02d}"
        
        # Calculer la taille du dossier
        total_size = 0
        file_count = 0
        
        try:
            for file_path in raw_dir.rglob("*"):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
                    file_count += 1
        except:
            pass
        
        size_mb = total_size / (1024 * 1024)
        
        if file_count > 0:
            print(f'\r[{chars[i]}] Téléchargement... {time_str} | {size_mb:.1f}MB | {file_count} fichiers', 
                  end='', flush=True)
        else:
            print(f'\r[{chars[i]}] Téléchargement... {time_str} | Initialisation...', 
                  end='', flush=True)
        
        i = (i + 1) % len(chars)
        time.sleep(1.0)  # Vérifier chaque seconde

def main():
    global stop_animation
    
    print("TÉLÉCHARGEMENT AVEC MONITORING")
    print("=" * 45)
    
    # Créer les dossiers
    base_dir = Path(__file__).parent.parent
    raw_dir = base_dir / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    print("[OK] Dossiers créés")
    
    try:
        import kaggle
        print("[OK] Kaggle API trouvée")
        
        # Test d'authentification
        kaggle.api.authenticate()
        print("[OK] Authentification réussie")
        print("[INFO] Taille typique du dataset : ~150-300MB")
        print("[INFO] Temps estimé : 1-5 minutes selon votre connexion")
        print()
        
        # Démarrer le monitoring
        stop_animation = False
        monitor_thread = threading.Thread(target=monitor_download_folder, args=(raw_dir,))
        monitor_thread.daemon = True
        monitor_thread.start()
        
        start_time = time.time()
        
        # Téléchargement
        kaggle.api.dataset_download_files(
            "anthonytherrien/image-classification-dataset-32-classes",
            path=str(raw_dir),
            unzip=True
        )
        
        # Arrêter le monitoring
        stop_animation = True
        time.sleep(0.5)
        
        duration = time.time() - start_time
        print(f"\r[SUCCESS] Téléchargement terminé en {duration:.1f}s!" + " " * 30)
        
        # Analyse finale
        print("\n[ANALYSIS] Analyse du contenu téléchargé :")
        
        total_size = 0
        total_files = 0
        class_count = 0
        
        for item in raw_dir.rglob("*"):
            if item.is_file():
                total_size += item.stat().st_size
                total_files += 1
            elif item.is_dir() and item.parent == raw_dir:
                class_count += 1
                files_in_class = len([f for f in item.iterdir() if f.is_file()])
                print(f"  [CLASS] {item.name}/ ({files_in_class} images)")
        
        print(f"\n[SUMMARY] Résumé final :")
        print(f"  Classes détectées : {class_count}")
        print(f"  Total d'images : {total_files}")
        print(f"  Taille totale : {total_size / (1024*1024):.1f} MB")
        print(f"  Vitesse moyenne : {(total_size / (1024*1024)) / duration:.1f} MB/s")
        
        print("\n[NEXT] Prêt pour l'exploration!")
        return True
        
    except ImportError:
        print("[ERROR] Kaggle API non installée")
        print("[FIX] Exécutez: pip install kaggle")
        return False
    except Exception as e:
        stop_animation = True
        print(f"\r[ERROR] Erreur : {e}" + " " * 30)
        return False

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        stop_animation = True
        print("\n[STOP] Interrompu par l'utilisateur")