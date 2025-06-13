import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    pipeline,
    DataCollatorWithPadding
)
from datasets import load_dataset, DatasetDict
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import os
import json


def create_tokenize_function(tokenizer, text_column, label_column, max_length):
    """
    Crée une fonction de tokenisation pour éviter les problèmes de sérialisation
    """

    def tokenize_function(examples):
        tokenized = tokenizer(
            examples[text_column],
            truncation=True,
            max_length=max_length,
            padding=False
        )

        if label_column in examples:
            labels = examples[label_column]
            if isinstance(labels[0], str):
                label_map = {label: idx for idx, label in enumerate(set(labels))}
                tokenized["labels"] = [label_map[label] for label in labels]
            else:
                tokenized["labels"] = labels

        return tokenized

    return tokenize_function


class TextClassificationPipeline:
    """
    Pipeline complet pour la classification de texte multi-classes
    avec Hugging Face Transformers et PyTorch
    """

    def __init__(self, model_name: str = "distilbert-base-uncased", num_labels: int = None):
        """
        Initialise le pipeline de classification

        Args:
            model_name: Nom du modèle pré-entraîné (BERT, RoBERTa, DistilBERT, etc.)
            num_labels: Nombre de classes pour la classification
        """
        self.model_name = model_name
        self.num_labels = num_labels
        self.tokenizer = None
        self.model = None
        self.trainer = None
        self.label_names = []
        self.id2label = {}
        self.label2id = {}

    def explore_huggingface_models(self, task: str = "text-classification") -> List[str]:
        """
        Explore et recommande des modèles Transformer adaptés à la classification de texte

        Args:
            task: Type de tâche (text-classification par défaut)

        Returns:
            Liste des modèles recommandés
        """
        recommended_models = {
            "text-classification": [
                "distilbert-base-uncased",
                "bert-base-uncased",
                "roberta-base",
                "albert-base-v2",
                "xlnet-base-cased",
                "electra-base-discriminator"
            ]
        }

        models = recommended_models.get(task, recommended_models["text-classification"])

        print(f"Modèles recommandés pour {task}:")
        for i, model in enumerate(models, 1):
            print(f"{i}. {model}")

        return models

    def load_dataset_from_huggingface(self, dataset_name: str) -> DatasetDict:
        """
        Charge un dataset depuis Hugging Face Hub

        Args:
            dataset_name: Nom du dataset (ex: "ag_news", "yelp_review_full")

        Returns:
            Dataset chargé
        """
        print(f"Chargement du dataset: {dataset_name}")

        if dataset_name == "ag_news":
            dataset = load_dataset("ag_news")
            self.label_names = ["World", "Sports", "Business", "Sci/Tech"]
            self.num_labels = 4

        elif dataset_name == "yelp_review_full":
            dataset = load_dataset("yelp_review_full")
            self.label_names = ["1 star", "2 stars", "3 stars", "4 stars", "5 stars"]
            self.num_labels = 5

        else:
            dataset = load_dataset(dataset_name)

        if self.label_names:
            self.id2label = {i: label for i, label in enumerate(self.label_names)}
            self.label2id = {label: i for i, label in enumerate(self.label_names)}

        print(f"Dataset chargé avec succès!")
        print(f"Nombre de classes: {self.num_labels}")
        print(f"Classes: {self.label_names}")

        return dataset

    def initialize_model_and_tokenizer(self):
        """
        Initialise le tokenizer et le modèle pour la classification
        """
        print(f"Initialisation du modèle: {self.model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        if self.num_labels:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=self.num_labels,
                id2label=self.id2label,
                label2id=self.label2id
            )
        else:
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)

        print("Modèle et tokenizer initialisés avec succès!")

    def preprocess_data(self, dataset: DatasetDict, text_column: str = "text",
                        label_column: str = "label", max_length: int = 512) -> DatasetDict:
        """
        Préprocesse les données pour l'entraînement

        Args:
            dataset: Dataset à préprocesser
            text_column: Nom de la colonne contenant le texte
            label_column: Nom de la colonne contenant les labels
            max_length: Longueur maximale des séquences

        Returns:
            Dataset préprocessé
        """
        print("Préprocessing des données...")

        tokenize_fn = create_tokenize_function(
            self.tokenizer, text_column, label_column, max_length
        )

        tokenized_dataset = dataset.map(
            tokenize_fn,
            batched=True,
            remove_columns=dataset["train"].column_names
        )

        print("Préprocessing terminé!")
        print(f"Colonnes finales: {tokenized_dataset['train'].column_names}")
        return tokenized_dataset

    def compute_metrics(self, eval_pred) -> Dict[str, float]:
        """
        Calcule les métriques d'évaluation (précision, rappel, F1-score)

        Args:
            eval_pred: Prédictions et vraies valeurs

        Returns:
            Dictionnaire des métriques
        """
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)

        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='weighted'
        )
        accuracy = accuracy_score(labels, predictions)

        return {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }

    def fine_tune_model(self, train_dataset, eval_dataset,
                        output_dir: str = "./results",
                        num_train_epochs: int = 3,
                        per_device_train_batch_size: int = 16,
                        per_device_eval_batch_size: int = 16,
                        learning_rate: float = 2e-5):
        """
        Fine-tune le modèle sur le dataset sélectionné

        Args:
            train_dataset: Dataset d'entraînement
            eval_dataset: Dataset de validation
            output_dir: Répertoire de sortie
            num_train_epochs: Nombre d'époques
            per_device_train_batch_size: Taille de batch pour l'entraînement
            per_device_eval_batch_size: Taille de batch pour l'évaluation
            learning_rate: Taux d'apprentissage
        """
        print("Début du fine-tuning...")

        data_collator = DataCollatorWithPadding(
            tokenizer=self.tokenizer,
            padding=True,
            return_tensors="pt"
        )

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            learning_rate=learning_rate,
            weight_decay=0.01,
            logging_dir=f'{output_dir}/logs',
            logging_steps=100,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            report_to=None,
            dataloader_pin_memory=False,
            remove_unused_columns=True,
            dataloader_drop_last=False,
            dataloader_num_workers=0
        )

        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=self.compute_metrics,
            data_collator=data_collator,
        )

        self.trainer.train()

        print("Fine-tuning terminé!")

    def evaluate_model(self, test_dataset) -> Dict[str, float]:
        """
        Évalue les performances du modèle (précision, rappel, F1-score)

        Args:
            test_dataset: Dataset de test

        Returns:
            Dictionnaire des métriques d'évaluation
        """
        print("Évaluation du modèle...")

        if self.trainer is None:
            raise ValueError("Le modèle doit être entraîné avant l'évaluation")

        eval_results = self.trainer.evaluate(test_dataset)

        print("Résultats d'évaluation:")
        for metric, value in eval_results.items():
            if not metric.startswith('eval_'):
                continue
            metric_name = metric.replace('eval_', '')
            print(f"{metric_name.capitalize()}: {value:.4f}")

        return eval_results

    def save_model(self, save_path: str):
        """
        Sauvegarde le modèle fine-tuné

        Args:
            save_path: Chemin de sauvegarde
        """
        print(f"Sauvegarde du modèle dans: {save_path}")

        os.makedirs(save_path, exist_ok=True)

        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)

        metadata = {
            "model_name": self.model_name,
            "num_labels": self.num_labels,
            "label_names": self.label_names,
            "id2label": self.id2label,
            "label2id": self.label2id
        }

        with open(os.path.join(save_path, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        print("Modèle sauvegardé avec succès!")

    def load_model(self, model_path: str):
        """
        Charge un modèle fine-tuné sauvegardé

        Args:
            model_path: Chemin du modèle sauvegardé
        """
        print(f"Chargement du modèle depuis: {model_path}")

        with open(os.path.join(model_path, "metadata.json"), "r") as f:
            metadata = json.load(f)

        self.model_name = metadata["model_name"]
        self.num_labels = metadata["num_labels"]
        self.label_names = metadata["label_names"]
        self.id2label = metadata["id2label"]
        self.label2id = metadata["label2id"]

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)

        print("Modèle chargé avec succès!")

    def predict(self, texts: List[str]) -> List[Dict]:
        """
        Effectue des prédictions sur de nouveaux textes

        Args:
            texts: Liste de textes à classifier

        Returns:
            Liste des prédictions avec scores
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Le modèle doit être initialisé/chargé avant la prédiction")

        classifier = pipeline(
            "text-classification",
            model=self.model,
            tokenizer=self.tokenizer,
            return_all_scores=True
        )

        predictions = []
        for text in texts:
            result = classifier(text)
            predictions.append({
                "text": text,
                "predictions": result
            })

        return predictions

    def run_complete_pipeline(self, dataset_name: str, model_name: str = None,
                              save_path: str = "./fine_tuned_model"):
        """
        Exécute le pipeline complet de A à Z

        Args:
            dataset_name: Nom du dataset Hugging Face
            model_name: Nom du modèle (optionnel)
            save_path: Chemin de sauvegarde du modèle
        """
        print("=== DÉMARRAGE DU PIPELINE COMPLET ===")

        print("\n1. Exploration des modèles recommandés:")
        self.explore_huggingface_models()

        if model_name:
            self.model_name = model_name

        print(f"\n2. Chargement du dataset: {dataset_name}")
        dataset = self.load_dataset_from_huggingface(dataset_name)

        print("\n3. Initialisation du modèle et tokenizer:")
        self.initialize_model_and_tokenizer()

        print("\n4. Préprocessing des données:")
        tokenized_dataset = self.preprocess_data(dataset)

        print("\n5. Fine-tuning du modèle:")
        self.fine_tune_model(
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["test"]
        )

        print("\n6. Évaluation finale:")
        eval_results = self.evaluate_model(tokenized_dataset["test"])

        print("\n7. Sauvegarde du modèle:")
        self.save_model(save_path)

        print("\n=== PIPELINE TERMINÉ AVEC SUCCÈS ===")

        return eval_results


if __name__ == "__main__":
    pipeline = TextClassificationPipeline()

    results = pipeline.run_complete_pipeline(
        dataset_name="ag_news",
        model_name="distilbert-base-uncased",
        save_path="../models/text_classifier_hugging_face_model"
    )

    test_texts = [
        "Apple announces new iPhone with advanced AI capabilities",
        "Manchester United wins the championship",
        "Stock market reaches new heights amid economic recovery"
    ]

    predictions = pipeline.predict(test_texts)
    for pred in predictions:
        print(f"\nTexte: {pred['text']}")
        print("Prédictions:")
        for p in pred['predictions']:
            print(f"  {p['label']}: {p['score']:.4f}")