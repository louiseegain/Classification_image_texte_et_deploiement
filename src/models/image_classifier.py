"""
Modèle de classification d'images CNN
Auteur: [Votre Nom]
Date: 13 Juin 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import yaml
from pathlib import Path

class ImageClassifier(nn.Module):
    """Modèle de classification d'images basé sur ResNet"""
    
    def __init__(self, num_classes=32, architecture='resnet50', pretrained=True, freeze_backbone=True):
        super(ImageClassifier, self).__init__()
        
        self.num_classes = num_classes
        self.architecture = architecture
        
        # Charger le modèle de base
        if architecture == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            feature_dim = self.backbone.fc.in_features
            # Remplacer la dernière couche
            self.backbone.fc = nn.Linear(feature_dim, num_classes)
            
        elif architecture == 'resnet18':
            self.backbone = models.resnet18(pretrained=pretrained)
            feature_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(feature_dim, num_classes)
            
        elif architecture == 'efficientnet_b0':
            self.backbone = models.efficientnet_b0(pretrained=pretrained)
            feature_dim = self.backbone.classifier[1].in_features
            self.backbone.classifier[1] = nn.Linear(feature_dim, num_classes)
            
        else:
            raise ValueError(f"Architecture {architecture} non supportée")
        
        # Geler le backbone si demandé
        if freeze_backbone:
            self.freeze_backbone()
    
    def freeze_backbone(self):
        """Geler les paramètres du backbone pour le transfer learning"""
        if self.architecture in ['resnet50', 'resnet18']:
            # Geler toutes les couches sauf la dernière
            for name, param in self.backbone.named_parameters():
                if 'fc' not in name:
                    param.requires_grad = False
        elif self.architecture == 'efficientnet_b0':
            # Geler toutes les couches sauf le classifier
            for name, param in self.backbone.named_parameters():
                if 'classifier' not in name:
                    param.requires_grad = False
    
    def unfreeze_layers(self, num_layers=2):
        """Dégeler progressivement les dernières couches"""
        if self.architecture in ['resnet50', 'resnet18']:
            # Dégeler les dernières couches du ResNet
            layers_to_unfreeze = []
            if num_layers >= 1:
                layers_to_unfreeze.append('layer4')
            if num_layers >= 2:
                layers_to_unfreeze.append('layer3')
            
            for name, param in self.backbone.named_parameters():
                for layer_name in layers_to_unfreeze:
                    if layer_name in name:
                        param.requires_grad = True
    
    def forward(self, x):
        """Forward pass"""
        return self.backbone(x)
    
    def get_features(self, x):
        """Extraire les features avant la classification"""
        if self.architecture in ['resnet50', 'resnet18']:
            # Passer par toutes les couches sauf fc
            x = self.backbone.conv1(x)
            x = self.backbone.bn1(x)
            x = self.backbone.relu(x)
            x = self.backbone.maxpool(x)
            
            x = self.backbone.layer1(x)
            x = self.backbone.layer2(x)
            x = self.backbone.layer3(x)
            x = self.backbone.layer4(x)
            
            x = self.backbone.avgpool(x)
            x = torch.flatten(x, 1)
            
            return x
        else:
            # Pour EfficientNet, utiliser une approche similaire
            return self.backbone.features(x)

def load_config(config_path):
    """Charger la configuration depuis un fichier YAML"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def create_model_from_config(config_path=None):
    """Créer un modèle à partir d'un fichier de configuration"""
    
    if config_path is None:
        # Configuration par défaut
        config = {
            'model': {
                'architecture': 'resnet50',
                'num_classes': 32,
                'pretrained': True,
                'freeze_backbone': True
            }
        }
    else:
        config = load_config(config_path)
    
    model_config = config['model']
    
    model = ImageClassifier(
        num_classes=model_config['num_classes'],
        architecture=model_config['architecture'],
        pretrained=model_config['pretrained'],
        freeze_backbone=model_config['freeze_backbone']
    )
    
    return model, config

def count_parameters(model):
    """Compter les paramètres du modèle"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'frozen': total_params - trainable_params
    }

def test_model():
    """Tester le modèle avec des données factices"""
    print("TEST DU MODÈLE")
    print("=" * 30)
    
    # Créer le modèle
    model, config = create_model_from_config()
    
    # Statistiques du modèle
    params = count_parameters(model)
    print(f"Architecture : {config['model']['architecture']}")
    print(f"Nombre de classes : {config['model']['num_classes']}")
    print(f"Paramètres totaux : {params['total']:,}")
    print(f"Paramètres entraînables : {params['trainable']:,}")
    print(f"Paramètres gelés : {params['frozen']:,}")
    
    # Test avec des données factices
    print(f"\nTest avec des données factices...")
    batch_size = 4
    x = torch.randn(batch_size, 3, 224, 224)  # Format standard ImageNet
    
    model.eval()
    with torch.no_grad():
        output = model(x)
        predictions = torch.softmax(output, dim=1)
        predicted_classes = torch.argmax(predictions, dim=1)
    
    print(f"Input shape : {x.shape}")
    print(f"Output shape : {output.shape}")
    print(f"Predictions : {predicted_classes.tolist()}")
    print(f"Confidence max : {predictions.max(dim=1)[0].tolist()}")
    
    print(f"\n[SUCCESS] Modèle testé avec succès !")

if __name__ == "__main__":
    test_model()