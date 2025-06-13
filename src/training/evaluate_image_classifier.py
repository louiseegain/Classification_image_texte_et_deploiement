\
import os
import json
from pathlib import Path
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Assuming this script is in src/training, and image_classifier.py is in the same directory
try:
    from .image_classifier import get_model, load_image_dataset
except ImportError:
    # Fallback for direct execution if PYTHONPATH is not set up
    # This assumes 'src' is the parent of 'training' and is in sys.path or current execution path allows this
    from image_classifier import get_model, load_image_dataset


# --- Configuration ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR_RELATIVE = "data/external/Image_Classification_Dataset_32_Classes"
MODEL_SAVE_DIR_RELATIVE = "models/image_classifier_model"
RESULTS_METRICS_DIR_RELATIVE = "results/metrics"
RESULTS_PLOTS_DIR_RELATIVE = "results/plots"

MODEL_FILENAME = "image_classifier_best.pth"
CLASS_NAMES_FILENAME = "class_names.json"
EVAL_METRICS_FILENAME = "image_classifier_evaluation_metrics.json"
CONFUSION_MATRIX_FILENAME = "image_classifier_confusion_matrix.png"

BATCH_SIZE = 32 # Can be adjusted for evaluation
TRAIN_VAL_SPLIT_RATIO = 0.8 # Must match the split ratio used in training
RANDOM_SEED = 42 # Must match the seed used in training for consistent split

# --- Paths ---
DATA_PATH = PROJECT_ROOT / DATA_DIR_RELATIVE
MODEL_DIR_PATH = PROJECT_ROOT / MODEL_SAVE_DIR_RELATIVE
MODEL_FILE_PATH = MODEL_DIR_PATH / MODEL_FILENAME
CLASS_NAMES_PATH = MODEL_DIR_PATH / CLASS_NAMES_FILENAME

RESULTS_METRICS_PATH = PROJECT_ROOT / RESULTS_METRICS_DIR_RELATIVE
RESULTS_PLOTS_PATH = PROJECT_ROOT / RESULTS_PLOTS_DIR_RELATIVE

def main():
    print(f"Project Root: {PROJECT_ROOT}")
    RESULTS_METRICS_PATH.mkdir(parents=True, exist_ok=True)
    RESULTS_PLOTS_PATH.mkdir(parents=True, exist_ok=True)

    # --- Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load Class Names ---
    if not CLASS_NAMES_PATH.exists():
        print(f"Error: Class names file not found at {CLASS_NAMES_PATH}")
        return
    with open(CLASS_NAMES_PATH, 'r') as f:
        class_names = json.load(f)
    num_classes = len(class_names)
    print(f"Loaded {num_classes} class names.")

    # --- Load Model ---
    if not MODEL_FILE_PATH.exists():
        print(f"Error: Model file not found at {MODEL_FILE_PATH}")
        return
    
    model = get_model(num_classes=num_classes, pretrained=False) # pretrained=False as we load our fine-tuned weights
    model.load_state_dict(torch.load(MODEL_FILE_PATH, map_location=device))
    model.to(device)
    model.eval()
    print(f"Model '{MODEL_FILENAME}' loaded successfully.")

    # --- Data Transformations (should match validation transforms from training) ---
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # --- Load Full Dataset ---
    print(f"Loading full dataset from: {DATA_PATH}")
    # Use a basic transform initially for splitting, then apply specific transform to val_dataset if needed,
    # or ensure the loaded dataset for split is the one with val_transform.
    # load_image_dataset from image_classifier.py takes a transform argument.
    full_dataset = load_image_dataset(DATA_PATH, transform=val_transform)
    if not full_dataset:
        print("Failed to load dataset. Exiting.")
        return
    
    # --- Split Dataset to get Validation Set ---
    # Ensure the split is the same as in training
    total_size = len(full_dataset)
    train_size = int(TRAIN_VAL_SPLIT_RATIO * total_size)
    val_size = total_size - train_size

    torch.manual_seed(RANDOM_SEED) # Set seed for reproducible split
    # We only need the validation subset here.
    # The `random_split` function returns Subset objects.
    # The `full_dataset` was already loaded with `val_transform`.
    _, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    print(f"Validation set size: {len(val_dataset)} images")

    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    print("Validation DataLoader created.")

    # --- Perform Predictions ---
    all_preds = []
    all_labels = []
    
    print("Starting evaluation on the validation set...")
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # --- Calculate Metrics ---
    print("\\nCalculating metrics...")
    accuracy = accuracy_score(all_labels, all_preds)
    # Use zero_division=0 to prevent warnings and errors for classes with no predictions/labels in some folds/scenarios
    precision_weighted = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall_weighted = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1_weighted = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    report_dict = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True, zero_division=0)
    report_str = classification_report(all_labels, all_preds, target_names=class_names, zero_division=0)

    metrics_summary = {
        "model_name": MODEL_FILENAME,
        "dataset_evaluated": "Validation Set (recreated from training split)",
        "num_classes": num_classes,
        "total_validation_samples": len(all_labels),
        "accuracy": accuracy,
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
        "f1_score_weighted": f1_weighted,
        "classification_report": report_dict,
        "timestamp": torch.Timestamp().strftime('%Y-%m-%dT%H:%M:%S') if hasattr(torch, 'Timestamp') else str(Path.cwd()) # Fallback for older torch
    }
    
    print("\\n--- Evaluation Metrics ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision (Weighted): {precision_weighted:.4f}")
    print(f"Recall (Weighted): {recall_weighted:.4f}")
    print(f"F1-Score (Weighted): {f1_weighted:.4f}")
    print("\\nClassification Report:")
    print(report_str)

    # --- Save Metrics ---
    metrics_file_path = RESULTS_METRICS_PATH / EVAL_METRICS_FILENAME
    with open(metrics_file_path, 'w') as f:
        json.dump(metrics_summary, f, indent=4)
    print(f"\\nEvaluation metrics saved to: {metrics_file_path}")

    # --- Confusion Matrix ---
    print("\\nGenerating confusion matrix...")
    cm = confusion_matrix(all_labels, all_preds)
    
    plt.figure(figsize=(max(12, num_classes * 0.5), max(10, num_classes * 0.4)))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                annot_kws={"size": 8 if num_classes > 10 else 10}) # Adjust font size
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title(f'Confusion Matrix - {MODEL_FILENAME}', fontsize=14)
    plt.xticks(rotation=45, ha="right", fontsize=8 if num_classes > 10 else 10)
    plt.yticks(rotation=0, fontsize=8 if num_classes > 10 else 10)
    plt.tight_layout()
    
    cm_plot_path = RESULTS_PLOTS_PATH / CONFUSION_MATRIX_FILENAME
    plt.savefig(cm_plot_path)
    print(f"Confusion matrix plot saved to: {cm_plot_path}")
    plt.close()

    print("\\nEvaluation script finished.")

if __name__ == "__main__":
    # This basic setup allows direct execution.
    # For more complex projects, consider adding proper PYTHONPATH setup or running as a module.
    current_script_path = Path(__file__).resolve()
    project_root_for_sys_path = current_script_path.parents[2] # d:/Dev/DL/Classification_image_texte_et_deploiement
    src_path = project_root_for_sys_path / "src"
    
    import sys
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
        # print(f"Added {src_path} to sys.path for relative imports")

    # Re-check imports if they failed initially due to path issues
    try:
        from training.image_classifier import get_model, load_image_dataset
    except ImportError as e:
        print(f"Could not re-import dependencies after sys.path modification: {e}")
        print("Please ensure the script is run from the project root or src is in PYTHONPATH.")
        # Fallback to try without 'training.' if running from within 'training' directory and '.' is in path
        try:
            from image_classifier import get_model, load_image_dataset
            print("Successfully imported using fallback (image_classifier directly).")
        except ImportError:
            print("Fallback import also failed. Check your execution context and PYTHONPATH.")
            raise

    main()
