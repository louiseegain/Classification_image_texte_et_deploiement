"""
Script pour diagnostiquer l'état du téléchargement
"""
from pathlib import Path
import time

def check_download_status():
    """Vérifier l'état actuel du téléchargement"""
    base_dir = Path(__file__).parent.parent
    raw_dir = base_dir / "data" / "raw"
    
    print("DIAGNOSTIC DU TÉLÉCHARGEMENT")
    print("=" * 40)
    print(f"Dossier surveillé : {raw_dir}")
    print()
    
    if not raw_dir.exists():
        print("[STATUS] Le dossier data/raw n'existe pas encore")
        return
    
    # Lister tout le contenu
    all_items = list(raw_dir.rglob("*"))
    
    print(f"[STATUS] {len(all_items)} éléments trouvés dans data/raw/")
    
    # Séparer fichiers et dossiers
    files = [item for item in all_items if item.is_file()]
    folders = [item for item in all_items if item.is_dir()]
    
    print(f"[FILES] {len(files)} fichiers")
    print(f"[FOLDERS] {len(folders)} dossiers")
    
    if files:
        print("\n[DETAIL] Fichiers trouvés :")
        total_size = 0
        for file_path in files[:10]:  # Montrer les 10 premiers
            size_mb = file_path.stat().st_size / (1024 * 1024)
            total_size += file_path.stat().st_size
            print(f"  {file_path.name} ({size_mb:.2f} MB)")
        
        if len(files) > 10:
            print(f"  ... et {len(files) - 10} autres fichiers")
        
        print(f"\n[SIZE] Taille totale actuelle : {total_size / (1024*1024):.1f} MB")
    
    if folders:
        print("\n[DETAIL] Dossiers trouvés :")
        for folder in folders:
            files_in_folder = len([f for f in folder.iterdir() if f.is_file()])
            print(f"  {folder.name}/ ({files_in_folder} fichiers)")
    
    # Chercher spécifiquement des fichiers ZIP
    zip_files = [f for f in files if f.suffix.lower() == '.zip']
    if zip_files:
        print(f"\n[ZIP] {len(zip_files)} fichiers ZIP trouvés :")
        for zip_file in zip_files:
            size_mb = zip_file.stat().st_size / (1024 * 1024)
            print(f"  {zip_file.name} ({size_mb:.1f} MB)")
        print("[INFO] Le fichier ZIP pourrait être en cours de décompression")
    
    # Estimation de progression
    if len(folders) >= 30:
        print("\n[PROGRESS] Téléchargement probablement terminé (30+ dossiers de classes)")
    elif len(folders) > 0:
        print(f"\n[PROGRESS] Téléchargement en cours ({len(folders)} dossiers de classes détectés)")
    elif len(files) > 0:
        print("\n[PROGRESS] Fichiers détectés, décompression possible en cours")
    else:
        print("\n[PROGRESS] Téléchargement en cours d'initialisation")

def monitor_continuous():
    """Surveiller en continu"""
    print("Surveillance en continu (Ctrl+C pour arrêter)")
    print("-" * 40)
    
    try:
        while True:
            check_download_status()
            print("\n" + "="*40)
            time.sleep(10)  # Vérifier toutes les 10 secondes
    except KeyboardInterrupt:
        print("\n[STOP] Surveillance arrêtée")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--monitor":
        monitor_continuous()
    else:
        check_download_status()