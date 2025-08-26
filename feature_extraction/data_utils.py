# =============================================================================
# Data Utilities for Deep Learning Pipeline
# =============================================================================

import os
import torch
import torchvision.transforms as transforms
import sys
import warnings
import pandas as pd
import numpy as np
from PIL import Image
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from timm.layers import SwiGLUPacked
from transformers import ViTForImageClassification, ViTConfig
from huggingface_hub import hf_hub_download

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Set Hugging Face token
os.environ['HF_TOKEN'] = '<your_token>'

def load_model(modeltype, magnification=20, custom_encoder_script=None, **kwargs):
    """
    Load foundation models for feature extraction.
    This is the critical function from the original data_utils.py
    
    Args:
        modeltype: Name of the model to load
        magnification: Magnification level (for some models)
        custom_encoder_script: Path to custom encoder loading script (optional)
        **kwargs: Additional arguments to pass to custom encoder load_model function
    """
    # If custom encoder script is provided, use it
    if custom_encoder_script and os.path.exists(custom_encoder_script):
        print(f"Loading custom encoder from: {custom_encoder_script}")
        try:
            # Import the custom encoder script
            import importlib.util
            spec = importlib.util.spec_from_file_location("custom_encoder", custom_encoder_script)
            custom_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(custom_module)
            
            # Call the load_model function from the custom script
            if hasattr(custom_module, 'load_model'):
                model = custom_module.load_model(modeltype, magnification, **kwargs)
                print(f"Successfully loaded custom {modeltype} model")
                return model
            else:
                print(f"Warning: Custom encoder script {custom_encoder_script} does not have load_model function")
        except Exception as e:
            print(f"Error loading custom encoder: {e}")
            print("Falling back to built-in encoders...")
    
    # Built-in encoder loading
    if modeltype == 'vit-large':
        modelpath = '/data/fuchs/neerajkumarvaid/vit_large_pathclassification/results/checkpoint-122648'
        config = ViTConfig.from_pretrained(modelpath)
        config.output_hidden_states = True
        model = ViTForImageClassification.from_pretrained(modelpath, config=config)
        model.cuda()
    elif modeltype == 'vit-base':
        if magnification == 20:
            modelpath = '/data/fuchs/neerajkumarvaid/vit_base_15epochs_pathclassification/results/checkpoint-57495'
        elif magnification == 40:
            modelpath = '/data/fuchs/neerajkumarvaid/vit_base_40x/results/checkpoint-6340'
        config = ViTConfig.from_pretrained(modelpath)
        config.output_hidden_states = True
        model = ViTForImageClassification.from_pretrained(modelpath, config=config)
        model.cuda()
    elif modeltype == 'uni':
        modelpath = "/data1/vanderbc/vanderbc/SSL_checkpoints/UNI_hf_original_model_checkpoint.pth"
        model = timm.create_model("vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True)
        model.load_state_dict(torch.load(modelpath), strict=True)
        model.cuda()
    elif modeltype == 'prov-gigapath':
        model = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)
        if model:
            print("Hugging Face prov-gigapath model loaded successfully.")
            model.cuda()
        else:
            print("Failed to load Hugging Face prov-gigapath model.")
    elif modeltype == 'virchow':
        print("Loading Virchow model")
        model = timm.create_model(
            "hf-hub:paige-ai/Virchow",
            pretrained=True,
            mlp_layer=SwiGLUPacked,
            act_layer=torch.nn.SiLU
        )
        model.eval()
        model.cuda()
        print("Virchow model loaded successfully.")
    elif modeltype == 'virchow2':
        print("Loading Virchow2 model")
        model = timm.create_model(
            "hf-hub:paige-ai/Virchow2",
            pretrained=True,
            mlp_layer=SwiGLUPacked,
            act_layer=torch.nn.SiLU
        )
        model.eval()
        model.cuda()
        print("Virchow2 model loaded successfully.")
    elif modeltype == 'h-optimus-0':
        print("Loading h-optimus-0 model")
        model = timm.create_model(
            "hf-hub:bioptimus/H-optimus-0", 
            pretrained=True, 
            init_values=1e-5, 
            dynamic_img_size=False
        )
        model.eval()
        model.cuda()
        print("h-optimus-0 model loaded successfully.")
    else:
        raise ValueError(f"Unsupported model type: {modeltype}")
    
    return model

def get_model_transform(modeltype, custom_encoder_script=None, **kwargs):
    """
    Get the appropriate transform for each model type.
    
    Args:
        modeltype: Name of the model
        custom_encoder_script: Path to custom encoder loading script (optional)
        **kwargs: Additional arguments to pass to custom encoder get_transform function
    """
    # If custom encoder script is provided, try to get transform from it
    if custom_encoder_script and os.path.exists(custom_encoder_script):
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location("custom_encoder", custom_encoder_script)
            custom_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(custom_module)
            
            # Call the get_transform function from the custom script
            if hasattr(custom_module, 'get_transform'):
                transform, feature_length = custom_module.get_transform(modeltype, **kwargs)
                print(f"Using custom transform for {modeltype}")
                return transform, feature_length
            else:
                print(f"Warning: Custom encoder script {custom_encoder_script} does not have get_transform function")
        except Exception as e:
            print(f"Error getting custom transform: {e}")
            print("Falling back to built-in transforms...")
    
    # Built-in transforms
    if modeltype == 'vit-large':
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        feature_length = 1024
    elif modeltype == 'vit-base':
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        feature_length = 768
    elif modeltype == 'uni':
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        feature_length = 1024
    elif modeltype == 'virchow':
        feature_length = 1280 * 2  # Virchow concatenates class token and pooled patch tokens
        config = resolve_data_config(model.pretrained_cfg, model=model)
        transform = create_transform(**config)
    elif modeltype == 'virchow2':
        feature_length = 1280 * 2  # Virchow2 also concatenates class token and pooled patch tokens
        transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
    elif modeltype == 'h-optimus-0':
        feature_length = 1536
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.707223, 0.578729, 0.703617), 
                std=(0.211883, 0.230117, 0.177517)
            )
        ])
    elif modeltype in ['prov-gigapath', 'gigapath_ft']:
        feature_length = 1536
        transform = transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
    else:
        raise ValueError(f"Unsupported model type: {modeltype}")
    
    return transform, feature_length

class FoundationModelLoader:
    """Class to load and manage foundation models for feature extraction."""
    
    def __init__(self, model_type, device='cuda', custom_encoder_script=None, **kwargs):
        self.model_type = model_type
        self.device = device
        self.model = None
        self.processor = None
        self.custom_encoder_script = custom_encoder_script
        self.kwargs = kwargs
        
    def load_model(self):
        """Load the foundation model using the critical load_model function."""
        try:
            # Use the critical load_model function from the original data_utils.py
            loaded_model = load_model(self.model_type, custom_encoder_script=self.custom_encoder_script, **self.kwargs)
            if loaded_model is not None:
                self.model = loaded_model
                print(f"Successfully loaded {self.model_type} model")
                return self.model
            else:
                print(f"Failed to load {self.model_type} model - returned None")
                return None
        except Exception as e:
            print(f"Error loading {self.model_type} model: {e}")
            return None
    
    def get_transform(self):
        """Get the appropriate transform for the model."""
        try:
            transform, feature_length = get_model_transform(self.model_type, custom_encoder_script=self.custom_encoder_script, **self.kwargs)
            return transform, feature_length
        except Exception as e:
            print(f"Error getting transform for {self.model_type}: {e}")
            return None, None

class DataProcessor:
    """Class to handle data processing and loading."""
    
    def __init__(self, manifest_path, slide_path, target_column, split_column="split"):
        self.manifest_path = manifest_path
        self.slide_path = slide_path
        self.target_column = target_column
        self.split_column = split_column
        self.data = None
        
    def load_manifest(self):
        """Load and validate the manifest file."""
        if not os.path.exists(self.manifest_path):
            raise FileNotFoundError(f"Manifest file not found: {self.manifest_path}")
        
        self.data = pd.read_csv(self.manifest_path)
        
        # Validate required columns
        required_columns = ['slide_name', self.target_column]
        missing_columns = [col for col in required_columns if col not in self.data.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns in manifest: {missing_columns}")
        
        # Clean slide names (remove .svs extension if present)
        self.data['slide_name'] = self.data['slide_name'].str.replace('.svs', '', regex=False)
        
        print(f"Loaded manifest with {len(self.data)} slides")
        return self.data
    
    def get_split_data(self, split_name):
        """Get data for a specific split."""
        if self.data is None:
            self.load_manifest()
        
        if self.split_column not in self.data.columns:
            raise ValueError(f"Split column '{self.split_column}' not found in manifest")
        
        split_data = self.data[self.data[self.split_column] == split_name].copy()
        print(f"Found {len(split_data)} slides for split '{split_name}'")
        return split_data
    
    def get_slide_path(self, slide_name):
        """Get the full path to a slide file."""
        # Try different extensions
        extensions = ['.svs', '.tif', '.tiff', '.ndpi']
        
        for ext in extensions:
            slide_file = os.path.join(self.slide_path, f"{slide_name}{ext}")
            if os.path.exists(slide_file):
                return slide_file
        
        # If no file found, return None
        return None
    
    def get_tcga_slide_path(self, slide_name):
        """
        Get the full path to a TCGA slide file using UUID-based directory structure.
        
        Args:
            slide_name: Name of the slide file (e.g., "TCGA-E7-A97Q-01Z-00-DX1.821AF545-5433-4D2C-AACE-EA206E15B13D.svs")
            
        Returns:
            Full path to the slide file, or None if not found
        """
        import glob
        
        # Try different extensions
        extensions = ['.svs', '.tif', '.tiff', '.ndpi']
        
        for ext in extensions:
            # Handle case where slide_name already has extension
            if slide_name.endswith(ext):
                file_name = slide_name
            else:
                file_name = f"{slide_name}{ext}"
            
            # Search pattern for UUID-based structure: svs/*/filename
            search_pattern = os.path.join(self.slide_path, '*', file_name)
            matching_files = glob.glob(search_pattern, recursive=True)
            
            if matching_files:
                return matching_files[0]
        
        # If no file found, return None
        return None
    
    def find_slide_path(self, slide_name, use_tcga_structure=False):
        """
        Find the full path to a slide file, with option for TCGA UUID-based structure.
        
        Args:
            slide_name: Name of the slide file
            use_tcga_structure: If True, use UUID-based directory search
            
        Returns:
            Full path to the slide file, or None if not found
        """
        if use_tcga_structure:
            return self.get_tcga_slide_path(slide_name)
        else:
            return self.get_slide_path(slide_name)

def load_checkpoint(checkpoint_path, model, device, model_type='slide'):
    """Load checkpoint correctly based on model type."""
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    try:
        if model_type == 'tile':
            # For tile model, load from 'tile_model' key
            model.load_state_dict(checkpoint['tile_model'], strict=True)
        else:
            # For slide model, load the state dict directly
            model.load_state_dict(checkpoint['slide_model'], strict=True)
        print(f"Successfully loaded {model_type} model from {checkpoint_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to load {model_type} model: {str(e)}")
    
    return model

def fix_state_dict(state_dict):
    """Fix state dict keys for DataParallel models."""
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k  # remove 'module.' prefix
        new_state_dict[name] = v
    return new_state_dict 