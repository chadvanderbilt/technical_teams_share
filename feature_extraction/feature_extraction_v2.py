#!/usr/bin/env python3
"""
Feature Extraction v2 - Separated from Label Storage

This script extracts features from whole slide images without storing labels.
Features are extracted once per slide and can be reused for multiple binary tasks.
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import openslide
from PIL import Image
import logging
import time
from tqdm import tqdm
import json
from datetime import datetime
import warnings

# Suppress TIFF warnings
warnings.filterwarnings("ignore", message="TIFFFetchNormalTag")

# Add the parent directory to the path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.data_utils import FoundationModelLoader, DataProcessor

def detect_feature_naming_pattern(feature_dir):
    """
    Detects the naming pattern of existing feature files in a directory.
    Returns a tuple (prefix, suffix) or None if no consistent pattern is found.
    """
    if not os.path.exists(feature_dir):
        return None

    # List all files in the directory
    files = [f for f in os.listdir(feature_dir) if os.path.isfile(os.path.join(feature_dir, f))]
    
    # Look for files with common feature naming patterns
    feature_files = [f for f in files if f.startswith('features_') and f.endswith('.pt')]
    
    if not feature_files:
        return None
    
    # Analyze the pattern of existing feature files
    for f in feature_files:
        # Remove 'features_' prefix and '.pt' suffix
        name_part = f[9:-3]  # Remove 'features_' and '.pt'
        
        # Check for different patterns
        if name_part.startswith('imgP-'):
            # Pattern: features_imgP-XXXXX.pt
            return ('img', '.pt')
        elif name_part.startswith('P-'):
            # Pattern: features_P-XXXXX.pt
            return ('', '.pt')
        elif name_part.startswith('img'):
            # Pattern: features_imgXXXXX.pt
            return ('img', '.pt')
        else:
            # Pattern: features_XXXXX.pt (no prefix)
            return ('', '.pt')
    
    # If no consistent pattern found, return None
    return None

def get_feature_file_path(feature_dir, DMP_ASSAY_ID, slide_name, pipeline_mode, naming_pattern):
    """
    Generates the feature file path based on the detected naming pattern.
    """
    if naming_pattern is None:
        # Fallback to default naming if no pattern is detected
        if pipeline_mode == 'impact':
            if DMP_ASSAY_ID and DMP_ASSAY_ID.startswith('P-'):
                return os.path.join(feature_dir, f"features_img{DMP_ASSAY_ID}.pt")
            else:
                return os.path.join(feature_dir, f"features_{slide_name}.pt")
        elif pipeline_mode == 'tcga':
            return os.path.join(feature_dir, f"features_{slide_name}.pt")
        else:
            return os.path.join(feature_dir, f"features_{slide_name}.pt")

    prefix, suffix = naming_pattern

    if pipeline_mode == 'impact':
        if DMP_ASSAY_ID and DMP_ASSAY_ID.startswith('P-'):
            # Use the detected pattern: features_{prefix}{DMP_ASSAY_ID}{suffix}
            return os.path.join(feature_dir, f"features_{prefix}{DMP_ASSAY_ID}{suffix}")
        else:
            return os.path.join(feature_dir, f"features_{prefix}{slide_name}{suffix}")
    elif pipeline_mode == 'tcga':
        return os.path.join(feature_dir, f"features_{prefix}{slide_name}{suffix}")
    else:
        return os.path.join(feature_dir, f"features_{prefix}{slide_name}{suffix}")

def setup_logging(log_level="INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('feature_extraction_v2.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def load_config(config_file):
    """Load configuration from shell script and environment variables."""
    config = {}
    import os
    # First, load all environment variables (these take precedence)
    for key, value in os.environ.items():
        config[key] = value
    # Then, load from config file if not already set by env
    with open(config_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('export ') and '=' in line:
                key, value = line[7:].split('=', 1)
                # Remove quotes if present
                if value.startswith('"') and value.endswith('"'):
                    value = value[1:-1]
                if key not in config:
                    config[key] = value
    return config

def get_optimal_batch_size(model, device, target_batch_size, max_memory_gb=70):
    """
    Dynamically adjust batch size based on model and GPU memory.
    Conservative approach to prevent OOM errors.
    
    Args:
        model: The loaded model
        device: Device to run model on
        target_batch_size: Requested batch size
        max_memory_gb: Maximum memory to use (conservative limit)
        
    Returns:
        Optimal batch size that fits in available memory
    """
    if device.type != 'cuda':
        print(f"Device {device.type} detected, using target batch size: {target_batch_size}")
        return target_batch_size
    
    try:
        # Get GPU memory info
        gpu_memory = torch.cuda.get_device_properties(device).total_memory / 1e9
        print(f"GPU memory available: {gpu_memory:.1f}GB")
        print(f"Target batch size: {target_batch_size}")
        print(f"Memory limit: {max_memory_gb}GB")
        
        # Start with a very conservative test batch size
        test_batch_size = min(target_batch_size, 4)  # Start with max 4
        
        # Test with small batch first to measure memory usage
        print(f"Testing with batch size: {test_batch_size}")
        
        # Create dummy input (assuming 224x224 images, adjust if different)
        dummy_input = torch.randn(test_batch_size, 3, 224, 224).to(device)
        
        # Ensure model is on the same device as input
        model_device = next(model.parameters()).device
        if model_device != device:
            print(f"Moving model from {model_device} to {device}")
            model = model.to(device)
        
        # Clear cache and measure memory usage
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        with torch.no_grad():
            _ = model(dummy_input)
        
        memory_used = torch.cuda.max_memory_allocated() / 1e9
        print(f"Test batch {test_batch_size} used {memory_used:.2f}GB")
        
        # Calculate optimal batch size with conservative safety margin
        available_memory = min(max_memory_gb, gpu_memory * 0.8)  # Use 80% of GPU memory max
        safety_factor = 0.6  # Conservative 60% of calculated capacity
        
        if memory_used > 0:
            calculated_batch_size = int((available_memory / memory_used) * test_batch_size * safety_factor)
            optimal_batch_size = min(calculated_batch_size, target_batch_size)
        else:
            optimal_batch_size = 1
        
        # Ensure minimum batch size of 1
        optimal_batch_size = max(1, optimal_batch_size)
        
        print(f"Calculated optimal batch size: {optimal_batch_size}")
        print(f"Memory usage per tile: {memory_used/test_batch_size:.2f}GB")
        
        return optimal_batch_size
        
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"OOM error during batch size calculation: {e}")
            print("Falling back to batch size 1")
            return 1
        else:
            print(f"Error during batch size calculation: {e}")
            print("Falling back to target batch size")
            return target_batch_size
    except Exception as e:
        print(f"Unexpected error during batch size calculation: {e}")
        print("Falling back to target batch size")
        return target_batch_size

def process_images_in_batches(images, model, batch_size, device):
    """
    Process images in batches for efficiency.
    
    Args:
        images: List of image tensors
        model: The loaded model
        batch_size: Batch size to use
        device: Device to run model on
        
    Returns:
        Torch tensor of features
    """
    if batch_size <= 1:
        print("Processing images one at a time (batch_size <= 1)")
        return process_images_single(images, model, device)
    
    print(f"Processing {len(images)} images in batches of {batch_size}")
    features = []
    
    # Process images in batches
    for i in range(0, len(images), batch_size):
        batch_end = min(i + batch_size, len(images))
        batch_images = images[i:batch_end]
        
        print(f"Processing batch {i//batch_size + 1}/{(len(images) + batch_size - 1)//batch_size} "
              f"({len(batch_images)} images)")
        
        # Stack images into a single batch tensor
        batch_tensor = torch.stack(batch_images).to(device)
        
        # Clear cache before processing batch
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        
        # Single forward pass for entire batch
        with torch.no_grad():
            model.eval()
            
            # Ensure model is on the same device as input
            model_device = next(model.parameters()).device
            if model_device != batch_tensor.device:
                print(f"Moving model from {model_device} to {batch_tensor.device}")
                model = model.to(batch_tensor.device)
            
            batch_output = model(batch_tensor)
        
        # Extract features from batch output
        batch_features = extract_features_from_output(batch_output)
        
        # Move to CPU but keep as tensor
        batch_features = batch_features.cpu()
        features.append(batch_features)
        
        # Progress update
        print(f"✓ Batch {i//batch_size + 1} completed, features shape: {batch_features.shape}")
    
    # Concatenate all batch results
    if len(features) > 0:
        return torch.cat(features, dim=0)
    else:
        # Return empty 2D tensor with unknown feature dimension
        return torch.empty(0, 0)

def process_images_single(images, model, device):
    """
    Fallback: Process images one at a time (original method).
    
    Args:
        images: List of image tensors
        model: The loaded model
        device: Device to run model on
        
    Returns:
        Torch tensor of features
    """
    print("Processing images one at a time...")
    features = []
    
    with torch.no_grad():
        model.eval()
        for image in tqdm(images, desc="Extracting features"):
            # Move to device and add batch dimension
            image = image.unsqueeze(0).to(device)
            
            # Extract features using the model
            output = model(image)
            
            # Get features (handle different output formats)
            feature = extract_features_from_output(output)
            features.append(feature.cpu())
    
    if len(features) > 0:
        return torch.cat(features, dim=0)
    else:
        # Return empty 2D tensor with unknown feature dimension
        return torch.empty(0, 0)

def extract_features_from_output(output):
    """
    Extract features from model output, handling different output formats.
    
    Args:
        output: Model output (can be tensor, dict, or other formats)
        
    Returns:
        Feature tensor (always 2D: batch_size x feature_dim)
    """
    if hasattr(output, 'last_hidden_state'):
        features = output.last_hidden_state.mean(dim=1)  # Global average pooling
    elif hasattr(output, 'hidden_states'):
        features = output.hidden_states[-1].mean(dim=1)
    elif isinstance(output, dict) and 'x_norm_clstoken' in output:
        features = output['x_norm_clstoken']  # DINOv3 specific
    else:
        features = output
    
    # Ensure features are always 2D (batch_size, feature_dim)
    if features.dim() == 1:
        features = features.unsqueeze(0)  # Add batch dimension
    elif features.dim() > 2:
        # Flatten any extra dimensions beyond batch and feature
        features = features.flatten(start_dim=1)
    
    return features

def extract_tile_features(slide_path, tile_coords, model_loader, tile_size, device, batch_size, expected_slide_name=None):
    """
    Extract features from tiles in a slide.
    
    Args:
        slide_path: Path to the slide file
        tile_coords: DataFrame with tile coordinates
        model_loader: Foundation model loader
        tile_size: Size of tiles
        device: Device to run model on
        batch_size: Batch size for processing tiles
        expected_slide_name: Expected slide name for the result dictionary
        
    Returns:
        Dictionary mapping slide name to features
    """
    try:
        # Open slide
        try:
            slide = openslide.OpenSlide(slide_path)
        except Exception as e:
            print(f"Error opening slide {slide_path}: {e}")
            return {}
        
        # Filter tiles for this slide
        # slide_path might be just filename, but tiling file has full path
        # Extract just the filename from slide_path for matching
        slide_filename = os.path.basename(slide_path)
        slide_tiles = tile_coords[tile_coords['slide'].str.contains(slide_filename, na=False)]
        
        if len(slide_tiles) == 0:
            slide.close()
            return {}
        
        # Extract tiles and convert to images
        images = []
        valid_tiles = []
        
        # Get transform for the model
        transform, _ = model_loader.get_transform()
        if transform is None:
            print(f"Error: Could not get transform for {slide_path}")
            return {}
        
        for idx, tile_info in tqdm(slide_tiles.iterrows(), total=len(slide_tiles), desc=f"Processing tiles for {os.path.basename(slide_path)}"):
            x, y = tile_info['x'], tile_info['y']
            
            # Extract tile
            tile = slide.read_region((x, y), 0, (tile_size, tile_size))
            tile = tile.convert('RGB')
            
            # Apply transform to convert PIL image to tensor
            processed_tile = transform(tile)
            images.append(processed_tile)
            valid_tiles.append(tile_info)
        
        slide.close()
        
        if not images:
            return {}
        
        # Extract features using the model directly
        print(f"Starting feature extraction for {slide_path}")
        
        if model_loader.model is None:
            print(f"Error: Model is None for {slide_path}")
            return {}
        
        # Process images and extract features with dynamic batch size adjustment
        print(f"Processing {len(images)} images...")
        
        # Get optimal batch size for this model and device
        optimal_batch_size = get_optimal_batch_size(
            model_loader.model, 
            model_loader.device, 
            batch_size,  # Use the batch_size parameter that was previously ignored
            max_memory_gb=70  # Conservative memory limit
        )
        
        print(f"Using optimal batch size: {optimal_batch_size} (requested: {batch_size})")
        
        # Process images in batches for efficiency
        features = process_images_in_batches(
            images, 
            model_loader.model, 
            optimal_batch_size, 
            model_loader.device
        )
        
        if isinstance(features, torch.Tensor) and features.numel() > 0:
            print(f"Features extracted successfully, shape: {features.shape}")
        else:
            print(f"Error: No features extracted for {slide_path}")
            return {}
        
        # Create result dictionary
        if expected_slide_name is not None:
            slide_name = expected_slide_name
        else:
            slide_name = valid_tiles[0]['slide'].split('/')[-1].replace('.svs', '')  # Extract filename from full path
        
        result = {
            slide_name: {
                'features': features,
                'tile_coords': valid_tiles,
                'metadata': {
                    'slide_path': slide_path,
                    'num_tiles': len(valid_tiles),
                    'tile_size': tile_size,
                    'feature_dim': features.shape[1] if len(features.shape) > 1 else features.shape[0],
                    'extraction_time': datetime.now().isoformat(),
                    'encoder_model': model_loader.model_type
                }
            }
        }
        
        return result
        
    except Exception as e:
        print(f"Error processing slide {slide_path}: {e}")
        return {}

def process_slides_for_feature_extraction(manifest_path, slide_path, feature_dir, 
                                        model_loader, tile_size, device, config, tile_coords_csv, batch_size):
    """
    Process slides for feature extraction with flexible naming pattern detection.
    
    Args:
        manifest_path: Path to manifest CSV
        slide_path: Directory containing slide files
        feature_dir: Directory to save features
        model_loader: Foundation model loader
        tile_size: Size of tiles
        device: Device to run model on
        config: Configuration dictionary
        tile_coords_csv: Path to tile coordinates CSV
        batch_size: Batch size for processing tiles
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Processing slides for feature extraction")
    logger.info(f"Manifest: {manifest_path}")
    logger.info(f"Slide directory: {slide_path}")
    logger.info(f"Feature directory: {feature_dir}")
    logger.info(f"Tile coordinates: {tile_coords_csv}")
    
    # Load manifest
    manifest = pd.read_csv(manifest_path)
    logger.info(f"Loaded manifest with {len(manifest)} slides")
    
    # Load tile coordinates
    all_tile_coords = pd.read_csv(tile_coords_csv)
    logger.info(f"Loaded {len(all_tile_coords)} tile coordinates from {tile_coords_csv}")
    
    # Debug: Show some examples of slide names in tile coordinates
    if len(all_tile_coords) > 0:
        sample_slides = all_tile_coords['slide'].head(3).tolist()
        logger.info(f"Sample slide paths in tile coords: {sample_slides}")
    
    # Create feature directory if it doesn't exist
    os.makedirs(feature_dir, exist_ok=True)
    logger.info(f"Created feature directory: {feature_dir}")
    
    # Detect existing naming pattern in feature directory
    naming_pattern = detect_feature_naming_pattern(feature_dir)
    logger.info(f"Detected naming pattern: {naming_pattern}")
    
    # Create metadata tracking
    metadata = {
        'slides_processed': [],
        'slides_skipped': [],
        'errors': [],
        'naming_pattern': naming_pattern
    }
    
    metadata_file = os.path.join(feature_dir, 'feature_extraction_metadata.json')
    
    # Process each slide
    for idx, row in tqdm(manifest.iterrows(), total=len(manifest), desc="Processing slides"):
        slide_name = None
        pipeline_mode = config.get('PIPELINE_MODE', 'default')
        
        # Get DMP_ASSAY_ID for IMPACT data (always try to get it)
        DMP_ASSAY_ID = row.get('DMP_ASSAY_ID', None)
        
        if pipeline_mode == 'impact':
            # For IMPACT data, use file_name or slideid to find the slide file
            if 'file_name' in row and pd.notna(row['file_name']):
                slide_name = str(row['file_name'])
            elif 'slideid' in row and pd.notna(row['slideid']):
                slide_name = str(row['slideid'])
            else:
                # Fallback to DMP_ASSAY_ID if no file_name/slideid
                slide_name = str(DMP_ASSAY_ID)
        elif pipeline_mode == 'tcga':
            # TCGA datasets typically use 'slide_name' or 'slideid'
            if 'slide_name' in row and pd.notna(row['slide_name']):
                slide_name = str(row['slide_name'])
            elif 'slideid' in row and pd.notna(row['slideid']):
                slide_name = str(row['slideid'])
        else:
            # Default mode - try common column names
            for col_name in ['slide_name', 'file_name', 'slideid', 'slide_id']:
                if col_name in row and pd.notna(row[col_name]):
                    slide_name = str(row[col_name])
                    break
        
        if slide_name is None:
            slide_name = f"slide_{idx}"
            logger.warning(f"No slide name found in row {idx} for {pipeline_mode} mode, using default: {slide_name}")
        
        logger.debug(f"Processing slide: {slide_name}")
        
        # Find slide file
        slide_file = None
        
        # First try the slide name as-is (in case it already has extension)
        potential_file = os.path.join(slide_path, slide_name)
        logger.debug(f"Checking for file: {potential_file}")
        if os.path.exists(potential_file):
            slide_file = potential_file
            logger.debug(f"Found slide file: {slide_file}")
        else:
            # Try adding extensions
            for ext in ['.svs', '.tif', '.tiff', '.ndpi']:
                potential_file = os.path.join(slide_path, f"{slide_name}{ext}")
                logger.debug(f"Checking for file: {potential_file}")
                if os.path.exists(potential_file):
                    slide_file = potential_file
                    logger.debug(f"Found slide file: {slide_file}")
                    break
        
        if slide_file is None:
            logger.warning(f"Could not find slide file for {slide_name}")
            metadata['slides_skipped'].append({
                'slide_name': slide_name,
                'reason': 'file_not_found'
            })
            continue
        
        # Check if features already exist using detected naming pattern
        feature_file = get_feature_file_path(feature_dir, DMP_ASSAY_ID, slide_name, pipeline_mode, naming_pattern)
        logger.info(f"🔍 IMPACT mode - DMP_ASSAY_ID: {DMP_ASSAY_ID}, slide_name: {slide_name}, feature_file: {feature_file}")
        
        if os.path.exists(feature_file):
            logger.info(f"Features already exist for {slide_name}, skipping")
            metadata['slides_processed'].append({
                'slide_name': slide_name,
                'status': 'skipped_existing',
                'feature_file': feature_file
            })
            continue
        
        # For each slide, filter tile_coords for that slide
        # The tiling file has full slide paths, but we need to match by filename
        # Extract just the filename from the slide column for matching
        slide_basename = os.path.basename(slide_name)
        slide_tile_coords = all_tile_coords[all_tile_coords['slide'].str.contains(slide_basename, na=False)]
        
        if slide_tile_coords.empty:
            logger.warning(f"No tile coordinates found for slide {slide_name} in {tile_coords_csv}")
            logger.debug(f"Looking for slide name: {slide_basename}")
            logger.debug(f"Available slide names in tile coords: {all_tile_coords['slide'].head().tolist()}")
            
            # Try alternative matching - look for slides that contain the slide name
            # This handles cases where the manifest has a different naming convention
            alternative_matches = all_tile_coords[all_tile_coords['slide'].str.contains(slide_name.replace('.svs', ''), na=False)]
            if not alternative_matches.empty:
                logger.info(f"Found {len(alternative_matches)} tile coordinates using alternative matching")
                slide_tile_coords = alternative_matches
            else:
                metadata['slides_skipped'].append({
                    'slide_name': slide_name,
                    'reason': 'no_tile_coords'
                })
                continue
        
        try:
            # Extract features
            result = extract_tile_features(slide_file, slide_tile_coords, model_loader, tile_size, device, batch_size, expected_slide_name=slide_name)
            
            if slide_name in result:
                # Save features - extract just the features tensor like UNI format
                features_data = result[slide_name]
                features_tensor = features_data['features']  # Extract just the features tensor
                
                if config.get('FEATURE_COMPRESSION', True):
                    # Save with compression
                    torch.save(features_tensor, feature_file, _use_new_zipfile_serialization=True)
                else:
                    # Save without compression
                    torch.save(features_tensor, feature_file)
                
                logger.info(f"Saved features for {slide_name}")
                metadata['slides_processed'].append({
                    'slide_name': slide_name,
                    'status': 'success',
                    'feature_file': feature_file,
                    'num_tiles': features_data['metadata']['num_tiles'],
                    'feature_dim': features_data['metadata']['feature_dim']
                })
            else:
                logger.warning(f"No features extracted for {slide_name}")
                metadata['slides_skipped'].append({
                    'slide_name': slide_name,
                    'reason': 'no_features_extracted'
                })
                
        except Exception as e:
            logger.error(f"Error processing slide {slide_name}: {e}")
            metadata['errors'].append({
                'slide_name': slide_name,
                'error': str(e)
            })
    
    # Save metadata
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Feature extraction completed. Processed: {len(metadata['slides_processed'])}, Skipped: {len(metadata['slides_skipped'])}")
    return metadata

def main():
    parser = argparse.ArgumentParser(description='Extract features from slides (v2)')
    parser.add_argument('--config', type=str, required=True, help='Configuration file')
    parser.add_argument('--manifest', type=str, help='Path to manifest CSV')
    parser.add_argument('--slide-path', type=str, help='Directory containing slide files')
    parser.add_argument('--feature-dir', type=str, help='Directory to save features')
    parser.add_argument('--encoder-model', type=str, help='Foundation model to use')
    parser.add_argument('--tile-size', type=int, help='Size of tiles')
    parser.add_argument('--batch-size', type=int, default=150, help='Batch size for feature extraction')
    parser.add_argument('--log-level', type=str, default='INFO', help='Logging level')
    parser.add_argument('--tile-coords-csv', type=str, required=True, help='Tile coordinates CSV file (from tiling.py)')
    parser.add_argument('--custom-encoder-script', type=str, help='Path to custom encoder script (optional)')
    parser.add_argument('--checkpoint-path', type=str, help='Path to model checkpoint (for custom encoders)')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_level)
    
    # Load configuration
    config = load_config(args.config)
    
    # Use command line arguments or fall back to config
    manifest_path = args.manifest or config.get('MANIFEST_PATH')
    slide_path = args.slide_path or config.get('SLIDE_PATH')
    feature_dir = args.feature_dir or os.path.join(config.get('FEATURES_DIR', ''), config.get('ENCODER_MODEL', 'prov-gigapath'))
    encoder_model = args.encoder_model or config.get('ENCODER_MODEL', 'prov-gigapath')
    tile_size = args.tile_size or int(config.get('TILE_SIZE', 512))
    batch_size = args.batch_size or int(config.get('BATCH_SIZE', 256))
    
    # Use custom encoder script from argument if provided
    custom_encoder_script = args.custom_encoder_script or config.get('CUSTOM_ENCODER_SCRIPT')
    config['CUSTOM_ENCODER_SCRIPT'] = custom_encoder_script
    
    # Validate inputs
    if not manifest_path:
        raise ValueError("Manifest path not specified")
    if not slide_path:
        raise ValueError("Slide path not specified")
    if not feature_dir:
        raise ValueError("Feature directory not specified")
    
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")
    if not os.path.exists(slide_path):
        raise FileNotFoundError(f"Slide directory not found: {slide_path}")
    
    tile_coords_csv = args.tile_coords_csv or config.get('TILE_COORDS_CSV')
    if not tile_coords_csv or not os.path.exists(tile_coords_csv):
        raise FileNotFoundError(f"Tile coordinates CSV not found: {tile_coords_csv}")

    logger.info(f"Manifest: {manifest_path}")
    logger.info(f"Slide directory: {slide_path}")
    logger.info(f"Feature directory: {feature_dir}")
    logger.info(f"Encoder model: {encoder_model}")
    logger.info(f"Tile size: {tile_size}")
    logger.info(f"Batch size: {batch_size}")
    
    # Create feature directory if it doesn't exist
    os.makedirs(feature_dir, exist_ok=True)
    logger.info(f"Created feature directory: {feature_dir}")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load foundation model
    custom_encoder_script = config.get('CUSTOM_ENCODER_SCRIPT')
    checkpoint_path = args.checkpoint_path or config.get('CHECKPOINT_PATH')
    logger.info(f"Custom encoder script: {custom_encoder_script}")
    logger.info(f"Checkpoint path: {checkpoint_path}")
    
    # Prepare kwargs for custom encoder
    encoder_kwargs = {}
    if checkpoint_path:
        encoder_kwargs['checkpoint_path'] = checkpoint_path
    # if encoder_model contains "TCGA" , replace it with "vitb_tcga_1024b_125000iter"
    if "TCGA" in encoder_model:
        encoder_model = 'vitb_tcga_1024b_125000iter'

    model_loader = FoundationModelLoader(encoder_model, device, custom_encoder_script, **encoder_kwargs)
    model = model_loader.load_model()
    
    if model is None:
        raise RuntimeError(f"Failed to load model: {encoder_model}")
    
    # Debug: Check if model is stored in model_loader
    logger.info(f"Model loaded: {model is not None}")
    logger.info(f"model_loader.model is None: {model_loader.model is None}")
    logger.info(f"model_loader.model type: {type(model_loader.model)}")
    if model_loader.model is not None:
        logger.info(f"model_loader.model device: {next(model_loader.model.parameters()).device}")
    
    # Process slides for feature extraction
    metadata = process_slides_for_feature_extraction(
        manifest_path, slide_path, feature_dir,
        model_loader, tile_size, device, config, tile_coords_csv, batch_size
    )
    
    logger.info("Feature extraction v2 completed successfully")
    logger.info(f"Features saved to: {feature_dir}")
    logger.info(f"Metadata saved to: {os.path.join(feature_dir, 'feature_extraction_metadata.json')}")

if __name__ == "__main__":
    main() 