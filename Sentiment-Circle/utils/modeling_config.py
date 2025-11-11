import json
import os
import torch.nn as nn
import torch
from .modeling_classifier import LinearLayer, MLP2Layer, ContrastiveClassifier
from typing import Dict, List, Optional

class LinearLayerConfig:
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        dropout: float = 0.1,
        layer: int = None,
        meta: dict = None,
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.layer = layer
        self.meta = meta or {}
        
    def to_dict(self):
        return {
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'dropout': self.dropout,
            'layer': self.layer,
            'meta': self.meta,
        }
    
    @classmethod
    def from_dict(cls, config_dict):
        return cls(
            output_dim=config_dict['output_dim'],
            dropout=config_dict['dropout'],
            meta=config_dict.get('meta', {}),
        )
        
    def save_pretrained(self, save_path: str, classifier_name: str):
        config_dict = self.to_dict()
        config_dict['classifier_name'] = classifier_name
        save_path = os.path.join(save_path, f"linear_layer:{self.layer}_dim:{self.output_dim}", f"{classifier_name}.json")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(config_dict, f, indent=4)

class MLP2LayerConfig:
    def __init__(
        self,
        input_dim: int,
        intermediate_dim: int,
        bottleneck_dim: int,
        output_dim: int,
        dropout: float = 0.1,
        layer: int = None,
        meta: dict = None,
    ):  
        self.input_dim = input_dim
        self.intermediate_dim = intermediate_dim
        self.bottleneck_dim = bottleneck_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.layer = layer
        self.meta = meta or {}
    
    def to_dict(self):
        return {
            'input_dim': self.input_dim,
            'intermediate_dim': self.intermediate_dim,
            'bottleneck_dim': self.bottleneck_dim,
            'output_dim': self.output_dim,
            'dropout': self.dropout,
            'layer': self.layer,
            'meta': self.meta,
        }
    
    @classmethod
    def from_dict(cls, config_dict):
        return cls(
            intermediate_dim=config_dict['intermediate_dim'],
            bottleneck_dim=config_dict['bottleneck_dim'],
            output_dim=config_dict['output_dim'],
            dropout=config_dict['dropout'],
            meta=config_dict.get('meta', {}),
        )
        
    def save_pretrained(self, save_path: str, classifier_name: str):
        config_dict = self.to_dict()
        config_dict['classifier_name'] = classifier_name
        save_path = os.path.join(save_path, f"mlp2_layer:{self.layer}_dim:{self.output_dim}", f"{classifier_name}.json")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(config_dict, f, indent=4)
            
class ContrastiveClassifierConfig:
    def __init__(
        self,
        input_dim: int,
        intermediate_dim: int,
        output_dim: int,
        dropout: float = 0.1,
        layer: int = None,
        meta: dict = None,
    ):
        self.input_dim = input_dim
        self.intermediate_dim = intermediate_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.layer = layer
        self.meta = meta or {}
        
    def to_dict(self):
        return {
            'input_dim': self.input_dim,
            'intermediate_dim': self.intermediate_dim,
            'output_dim': self.output_dim,
            'dropout': self.dropout,
            'layer': self.layer,
            'meta': self.meta,
        }
    
    @classmethod
    def from_dict(cls, config_dict):
        return cls(
            input_dim=config_dict['input_dim'],
            intermediate_dim=config_dict['intermediate_dim'],
            output_dim=config_dict['output_dim'],
            dropout=config_dict['dropout'],
            meta=config_dict.get('meta', {}),
        )
        
    def save_pretrained(self, save_path: str, classifier_name: str):
        config_dict = self.to_dict()
        config_dict['classifier_name'] = classifier_name
        save_path = os.path.join(save_path, f"contrastive_logit_layer:{self.layer}_dim:{self.output_dim}", f"{classifier_name}.json")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(config_dict, f, indent=4)

def build_classifiers(classifier_configs: dict, model_config) -> (nn.ModuleDict, dict):
    """
    Factory to build classifier modules and their configs.

    Args:
        classifier_configs: Mapping from classifier name to its parameter dict.
        model_config: Model configuration with attributes hidden_size, num_hidden_layers, etc.
    Returns:
        modules: nn.ModuleDict of classifier modules.
        clf_configs: Dict of classifier configuration objects.
    """
    modules = nn.ModuleDict()
    clf_configs = {}

    for name, params in classifier_configs.items():
        params = params.copy()
        params.setdefault("name", name)
        ctype = params.get("type")
        if ctype is None:
            raise ValueError(f"Classifier {name} - 'type' is required in the config.")

        if ctype == 'linear':
            cfg = LinearLayerConfig(
                input_dim=model_config.hidden_size,
                output_dim=params["output_dim"],
                dropout=params.get("dropout", 0.1),
                layer=params.get("layer", model_config.num_hidden_layers - 1),
                meta={
                    "name": name,
                    "type": "linear",
                    "objective": params.get("objective", "regression"),
                    "distance": params.get("distance", "cosine"),
                },
            )
            module = LinearLayer(cfg)

        elif ctype == 'contrastive_logit':
            cfg = ContrastiveClassifierConfig(
                input_dim=model_config.hidden_size,
                intermediate_dim=params["intermediate_dim"],
                output_dim=params["output_dim"],
                dropout=params.get("dropout", 0.1),
                layer=params.get("layer", model_config.num_hidden_layers - 1),
                meta={
                    "name": name,
                    "type": "contrastive_logit",
                    "objective": params.get("objective", "contrastive_logit"),
                    "distance": params.get("distance", "cosine"),
                },
            )
            module = ContrastiveClassifier(cfg)

        elif ctype == 'mlp2':
            cfg = MLP2LayerConfig(
                input_dim=model_config.hidden_size,
                intermediate_dim=params["intermediate_dim"],
                bottleneck_dim=params["bottleneck_dim"],
                output_dim=params["output_dim"],
                dropout=params.get("dropout", 0.1),
                layer=params.get("layer", model_config.num_hidden_layers - 1),
                meta={
                    "name": name,
                    "type": "mlp2",
                    "objective": params.get("objective", "regression"),
                    "distance": params.get("distance", "cosine"),
                },
            )
            module = MLP2Layer(cfg)

        else:
            raise ValueError(f"Unknown classifier type: {ctype}")

        modules[name] = module
        clf_configs[name] = cfg

    return modules, clf_configs

def load_classifiers(classifier_configs: Dict, model_config: Dict, save_dir: List, classifier_freeze: List) -> nn.ModuleDict:
    """
    Builds classifiers and loads their weights from files.

    Args:
        classifier_configs: Mapping from classifier name to its parameter dict.
        model_config: Model configuration.
        save_dir: List of Directory where classifier weights are stored.

    Returns:
        ModuleDict of loaded classifier modules.
    """
    modules, _ = build_classifiers(classifier_configs, model_config)

    for name, classifier_path in zip(list(modules.keys()), save_dir):
        weight_path = classifier_path
        if not os.path.exists(weight_path):
            raise FileNotFoundError(f"Classifier weight file for {name} not found at {weight_path}")
        state = {} if not os.path.getsize(weight_path) else torch.load(weight_path, map_location="cpu")
        modules[name].load_state_dict(state)
        
        if name in classifier_freeze:
            for param in modules[name].parameters():
                param.requires_grad = False
            print(f"Classifier {name} is frozen and will not be trained.")

    return modules
