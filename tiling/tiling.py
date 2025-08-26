#!/usr/bin/env python3
"""
Tiling script for whole slide image processing.
Extracts tiles from whole slide images based on manifest specifications.
Uses the correct SlideTileExtractor implementation for proper tissue detection.
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import openslide
import cv2
from PIL import Image
import logging
import random


from skimage.morphology import binary_erosion, binary_dilation, label, dilation, square, skeletonize
from skimage.filters import threshold_otsu

def setup_logging(log_level="INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('tiling.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def load_config(config_file):
    """Load configuration from shell script."""
    config = {}
    with open(config_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('export ') and '=' in line:
                key, value = line[7:].split('=', 1)
                config[key] = value.strip('"')
    return config

# Constants from original implementation
msk_aperio_20x_mpp = 0.5
MAX_PIXEL_DIFFERENCE = 0.1  # difference must be within 10% of image size

def normalize_msk20x(slide):
    mpp = float(slide.properties[openslide.PROPERTY_NAME_MPP_X])
    mult = msk_aperio_20x_mpp/mpp
    level = 0
    return level, mult



# https://github.com/MSKCC-Computational-Pathology/slidereader/blob/master/modules/gen_coords/extract_molecular_annotations.py
def detect_marker(thumb, mult):
    ksize = int(max(1, mult))
    # ksize = 1
    img = cv2.GaussianBlur(thumb, (5,5), 0)
    hsv_origimg = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    # Extract marker
    black_marker = cv2.inRange(hsv_origimg, np.array([0, 0, 0]), np.array([180, 255, 125])) # black marker
    blue_marker = cv2.inRange(hsv_origimg, np.array([90, 30, 30]), np.array([130, 255, 255])) # blue marker
    green_marker = cv2.inRange(hsv_origimg, np.array([40, 30, 30]), np.array([90, 255, 255])) # green marker
    mask_hsv = cv2.bitwise_or(cv2.bitwise_or(black_marker, blue_marker), green_marker)
    mask_hsv = cv2.erode(mask_hsv, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize,ksize)))
    mask_hsv = cv2.dilate(mask_hsv, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize*3,ksize*3)))
    if np.count_nonzero(mask_hsv) > 0:
        return mask_hsv
    else:
        return None

def power2mpp(power):
    return msk_aperio_20x_mpp*20./power

def find_level(slide, res, patchsize=224):
    mpp = float(slide.properties[openslide.PROPERTY_NAME_MPP_X])
    downsample = res/mpp
    for i in range(slide.level_count)[::-1]:
        if abs(downsample / slide.level_downsamples[i] * patchsize - patchsize) < MAX_PIXEL_DIFFERENCE * patchsize or downsample > slide.level_downsamples[i]:
            level = i
            mult = downsample / slide.level_downsamples[level]
            break
    else:
        raise Exception('Requested resolution ({} mpp) is too high'.format(res))
    
    # Move mult to closest pixel
    mult = np.round(mult*patchsize)/patchsize
    if abs(mult*patchsize - patchsize) < MAX_PIXEL_DIFFERENCE * patchsize:
        mult = 1.
    return level, mult

def image2array(img):
    if img.__class__.__name__ == 'Image':
        if img.mode == 'RGB':
            img = np.array(img)
            r, g, b = np.rollaxis(img, axis=-1)
            img = np.stack([r, g, b], axis=-1)
        elif img.mode == 'RGBA':
            img = np.array(img)
            r, g, b, a = np.rollaxis(img, axis=-1)
            img = np.stack([r, g, b], axis=-1)
        else:
            sys.exit('Error: image is not RGB slide')
    img = np.uint8(img)
    return img

def is_sample(img, threshold=0.9, ratioCenter=0.1, wholeAreaCutoff=0.5, centerAreaCutoff=0.9):
    nrows, ncols = img.shape
    timg = cv2.threshold(img, 255*threshold, 1, cv2.THRESH_BINARY_INV)
    kernel = np.ones((5, 5), np.uint8)
    cimg = cv2.morphologyEx(timg[1], cv2.MORPH_CLOSE, kernel)
    crow = np.rint(nrows/2).astype(int)
    ccol = np.rint(ncols/2).astype(int)
    drow = np.rint(nrows*ratioCenter/2).astype(int)
    dcol = np.rint(ncols*ratioCenter/2).astype(int)
    centerw = cimg[crow-drow:crow+drow, ccol-dcol:ccol+dcol]
    if (np.count_nonzero(cimg) < nrows*ncols*wholeAreaCutoff) & (np.count_nonzero(centerw) < 4*drow*dcol*centerAreaCutoff):
        return False
    else:
        return True

def threshold(slide, size, res, maxres, mult=1):
    w = int(np.round(slide.dimensions[0]*1./(size*res/maxres))) * mult
    h = int(np.round(slide.dimensions[1]*1./(size*res/maxres))) * mult
    thumbnail = slide.get_thumbnail((w, h))
    thumbnail = thumbnail.resize((w, h))
    img_c = image2array(thumbnail)
    
    # Calculate std on color image
    std = np.std(img_c, axis=-1)
    
    # Image to bw
    img_g = cv2.cvtColor(img_c, cv2.COLOR_RGB2GRAY)
    
    # Detect markers
    marker = detect_marker(img_c, maxres/res*mult)
    
    # Otsu thresholding
    img_g = cv2.GaussianBlur(img_g, (5, 5), 0)
    
    # New Otsu with skimage and masked arrays
    if marker is not None:
        masked = np.ma.masked_array(img_g, marker > 0)
        t = threshold_otsu(masked.compressed())
        img_g = cv2.threshold(img_g, t, 255, cv2.THRESH_BINARY)[1]
    else:
        t, img_g = cv2.threshold(img_g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Exclude marker
    if marker is not None:
        img_g = cv2.subtract(~img_g, marker)
    else:
        img_g = 255 - img_g
    
    # Remove grays
    img_g[std < 5] = 0
    
    # Rescale
    if mult > 1:
        img_g = img_g.reshape(h//mult, mult, w//mult, mult).max(axis=(1, 3))
    
    return img_g, t

def remove_black_ink(img_g, th=50, delta=50):
    '''
    Image in gray scale
    Returns mask where ink is positive
    th=50 and delta=50 was chosen based on some slides
    '''
    dist = np.clip(img_g - float(th), 0, None)
    mask = dist < delta
    if mask.sum() > 0:
        mask_s = skeletonize(mask)
        d = int(np.round(0.1 * mask.sum() / mask_s.sum()))
        mask = dilation(mask, square(2*d+1))
        return mask
    else:
        return None

def filter_regions(img, min_size):
    l, n = label(img, return_num=True)
    for i in range(1, n+1):
        # Filter small regions
        if l[l == i].size < min_size:
            l[l == i] = 0
    return l

def add(overlap):
    return np.linspace(0, 1, overlap+1)[1:-1]

def add2offset(img, slide, patch_size, mpp, maxmpp):
    size_x = img.shape[1]
    size_y = img.shape[0]
    offset_x = np.floor((slide.dimensions[0]*1./(patch_size*mpp/maxmpp)-size_x)*(patch_size*mpp/maxmpp))
    offset_y = np.floor((slide.dimensions[1]*1./(patch_size*mpp/maxmpp)-size_y)*(patch_size*mpp/maxmpp))
    add_x = np.linspace(0, offset_x, size_x).astype(int)
    add_y = np.linspace(0, offset_y, size_y).astype(int)
    return add_x, add_y

def addoverlap(w, grid, overlap, patch_size, mpp, maxmpp, img, offset=0):
    o = (add(overlap)*(patch_size*mpp/maxmpp)).astype(int)
    ox, oy = np.meshgrid(o, o)
    connx = np.zeros(img.shape).astype(bool)
    conny = np.zeros(img.shape).astype(bool)
    connd = np.zeros(img.shape).astype(bool)
    connu = np.zeros(img.shape).astype(bool)
    connx[:, :-1] = img[:, 1:]
    conny[:-1, :] = img[1:, :]
    connd[:-1, :-1] = img[1:, 1:]
    connu[1:, :-1] = img[:-1, 1:] & (~img[1:, 1:] | ~img[:-1, :-1])
    connx = connx[w]
    conny = conny[w]
    connd = connd[w]
    connu = connu[w]
    extra = []
    for i, (x, y) in enumerate(grid):
        if connx[i]: 
            extra.extend(zip(o+x-offset, np.repeat(y, overlap-1)-offset))
        if conny[i]: 
            extra.extend(zip(np.repeat(x, overlap-1)-offset, o+y-offset))
        if connd[i]: 
            extra.extend(zip(ox.flatten()+x-offset, oy.flatten()+y-offset))
        if connu[i]: 
            extra.extend(zip(x+ox.flatten()-offset, y-oy.flatten()-offset))
    return extra

def get_tissue_trace(slide):
    maxmpp = float(slide.properties[openslide.PROPERTY_NAME_MPP_X])
    img, th = threshold(slide, 224, 0.5, maxmpp, 4)
    img[img > 0] = 1
    eroded = binary_erosion(img)
    img = img - eroded
    add_x, add_y = add2offset(img, slide, 224, 0.5, maxmpp)
    w = np.where(img > 0)
    grid = list(zip((w[1]*(224*0.5/maxmpp)+add_x[w[1]]).astype(int), 
                    (w[0]*(224*0.5/maxmpp)+add_y[w[0]]).astype(int)))
    offset = int(0.5 / maxmpp * 224 // 2)
    grid = [(x[0] + offset, x[1] + offset) for x in grid]
    return np.array(grid)

def make_sample_grid(slide, patch_size=224, mpp=0.5, power=None, min_cc_size=10, 
                    max_ratio_size=10, dilate=False, erode=False, prune=False, 
                    overlap=1, maxn=None, bmp=None, oversample=False, mult=1, centerpixel=False):
    '''
    Script that given an openslide object return a list of tuples
    in the form of (x,y) coordinates for patch extraction of sample patches.
    It has an erode option to make sure to get patches that are full of tissue.
    It has a prune option to check if patches are sample. It is slow.
    If bmp is given, it samples from within areas of the bmp that are nonzero.
    If oversample is True, it will downsample for full resolution regardless of what resolution is requested.
    mult is used to increase the resolution of the thumbnail to get finer tissue extraction
    '''
    if power:
        mpp = power2mpp(power)

    maxmpp = float(slide.properties[openslide.PROPERTY_NAME_MPP_X])
    if oversample:
        img, th = threshold(slide, patch_size, maxmpp, maxmpp, mult)
    else:
        img, th = threshold(slide, patch_size, mpp, maxmpp, mult)
    
    if bmp:
        bmplab = Image.open(bmp)
        thumbx, thumby = img.shape
        bmplab = bmplab.resize((thumby, thumbx), Image.ANTIALIAS)
        bmplab = np.array(bmplab)
        bmplab[bmplab > 0] = 1
        img = np.logical_and(img, bmplab)
    
    img = filter_regions(img, min_cc_size)
    img[img > 0] = 1
    if erode:
        img = binary_erosion(img)
    if dilate:
        img = binary_dilation(img)

    if oversample:
        add_x, add_y = add2offset(img, slide, patch_size, maxmpp, maxmpp)
    else:
        add_x, add_y = add2offset(img, slide, patch_size, mpp, maxmpp)
    
    # List of sample pixels
    w = np.where(img > 0)

    if oversample:
        offset = int(0.5 * patch_size * ((mpp/maxmpp) - 1))
        grid = list(zip((w[1]*(patch_size)+add_x[w[1]]-offset).astype(int),
                       (w[0]*(patch_size)+add_y[w[0]]-offset).astype(int)))
    else:
        grid = list(zip((w[1]*(patch_size*mpp/maxmpp)+add_x[w[1]]).astype(int),
                       (w[0]*(patch_size*mpp/maxmpp)+add_y[w[0]]).astype(int)))

    # Connectivity
    if overlap > 1:
        if oversample:
            extra = addoverlap(w, grid, overlap, patch_size, maxmpp, maxmpp, img, offset=offset)
            grid.extend(extra)
        else:
            extra = addoverlap(w, grid, overlap, patch_size, mpp, maxmpp, img)
            grid.extend(extra)

    # Center pixel offset
    if centerpixel:
        offset = int(mpp / maxmpp * patch_size // 2)
        grid = [(x[0] + offset, x[1] + offset) for x in grid]

    # Prune squares
    if prune:
        level, mult = find_level(slide, mpp, maxmpp)
        psize = int(patch_size*mult)
        truegrid = []
        for tup in grid:
            reg = slide.read_region(tup, level, (psize, psize))
            if mult != 1:
                reg = reg.resize((224, 224), Image.BILINEAR)
            reg = image2array(reg)
            if is_sample(reg, th/255, 0.2, 0.4, 0.5):
                truegrid.append(tup)
    else:
        truegrid = grid
    
    # Sample if maxn
    if maxn:
        truegrid = random.sample(truegrid, min(maxn, len(truegrid)))

    return truegrid

def process_manifest(manifest_path, slide_path, output_path, tile_size, overlap=0, 
                    min_cc_size=10, max_tiles=None, target_column=None):
    """
    Process all slides in the manifest using the correct SlideTileExtractor implementation.
    
    Args:
        manifest_path: Path to manifest CSV file
        slide_path: Directory containing slide files
        output_path: Directory to save tile coordinates
        tile_size: Size of tiles to extract
        overlap: Overlap between tiles
        min_cc_size: Minimum connected component size for tissue detection
        max_tiles: Maximum tiles per slide
        target_column: Target column name from manifest
    """
    # Load manifest
    try:
        manifest = pd.read_csv(manifest_path)
    except pd.errors.ParserError as e:
        print(f"Error parsing CSV file {manifest_path}: {e}")
        print("Attempting to read with error handling...")
        
        try:
            manifest = pd.read_csv(manifest_path, error_bad_lines=False, warn_bad_lines=True)
            print(f"Successfully loaded manifest with error handling. Shape: {manifest.shape}")
        except Exception as e2:
            print(f"Failed to load manifest even with error handling: {e2}")
            try:
                manifest = pd.read_csv(manifest_path, engine='python', error_bad_lines=False)
                print(f"Successfully loaded manifest with python engine. Shape: {manifest.shape}")
            except Exception as e3:
                print(f"All attempts to load manifest failed: {e3}")
                raise Exception(f"Cannot parse manifest file {manifest_path}. Please check the file format.")
    
    # Handle IMPACT data structure - use DMP_ASSAY_ID as slide_name
    if 'DMP_ASSAY_ID' in manifest.columns and 'slide_name' not in manifest.columns:
        manifest['slide_name'] = manifest['DMP_ASSAY_ID']
        print(f"Using DMP_ASSAY_ID as slide_name for {len(manifest)} slides")
    
    if target_column:
        manifest = manifest[['slide_name', target_column]]
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Process each slide using the correct implementation
    out_20x = []
    out_40x = []
    missed_slides = []
    
    for i, row in manifest.iterrows():
        print(f'[{i+1}/{len(manifest)}]', end='\n')
        try:
            # Find slide file with UUID-based structure support
            slide_file = None
            slide_name = row.slide_name
            
            # First try direct path
            for ext in ['.svs', '.tif', '.tiff', '.ndpi']:
                potential_file = os.path.join(slide_path, f"{slide_name}{ext}")
                if os.path.exists(potential_file):
                    slide_file = potential_file
                    break
            
            # For CPTAC mode, try the subdirectory structure with spaces
            if slide_file is None and 'CPTAC' in slide_path:
                import glob
                for ext in ['.svs', '.tif', '.tiff', '.ndpi']:
                    if slide_name.endswith(ext):
                        file_name = slide_name
                    else:
                        file_name = f"{slide_name}{ext}"
                    
                    # Search pattern for CPTAC structure: svs/*/AML/filename
                    search_pattern = os.path.join(slide_path, '*', '*', file_name)
                    matching_files = glob.glob(search_pattern, recursive=True)
                    
                    if matching_files:
                        slide_file = matching_files[0]
                        break
            
            # For TCGA mode, try UUID-based directory structure
            if slide_file is None and 'TCGA' in slide_path:
                import glob
                for ext in ['.svs', '.tif', '.tiff', '.ndpi']:
                    if slide_name.endswith(ext):
                        file_name = slide_name
                    else:
                        file_name = f"{slide_name}{ext}"
                    
                    # Search pattern for UUID-based structure: svs/*/filename
                    search_pattern = os.path.join(slide_path, '*', file_name)
                    matching_files = glob.glob(search_pattern, recursive=True)
                    
                    if matching_files:
                        slide_file = matching_files[0]
                        break
            
            if slide_file is None:
                print(f"Slide {slide_name} not found in {slide_path}")
                missed_slides.append(slide_name)
                continue
            
            slide = openslide.OpenSlide(slide_file)
            
            # Use the correct make_sample_grid function with proper parameters
            # For 20x slides, use mpp=0.5, for 40x slides, use mpp=0.25
            try:
                resolution = slide.properties[openslide.PROPERTY_NAME_OBJECTIVE_POWER]
                if resolution == '20':
                    mpp = 0.5
                    # Derive 20x tile size from 40x input: 20x needs smaller tiles for equivalent tissue area
                    effective_tile_size = tile_size // 2
                    print(f"Processing 20x slide {slide_name} with mpp={mpp}, tile_size={effective_tile_size} (derived from 40x tile_size={tile_size})")
                elif resolution == '40':
                    mpp = 0.25
                    effective_tile_size = tile_size
                    print(f"Processing 40x slide {slide_name} with mpp={mpp}, tile_size={effective_tile_size}")
                else:
                    print(f"Slide {slide_name} has unknown resolution: {resolution}, defaulting to 40x (mpp=0.25)")
                    mpp = 0.25
                    effective_tile_size = tile_size
            except KeyError:
                print(f"Warning: Slide {slide_name} missing objective power property. Defaulting to 40x (mpp=0.25)")
                mpp = 0.25
                effective_tile_size = tile_size
            
            # Extract tiles using the correct implementation
            tile_coords = make_sample_grid(
                slide, 
                patch_size=effective_tile_size, 
                mpp=mpp, 
                min_cc_size=min_cc_size, 
                overlap=overlap, 
                maxn=None,  # No limit on number of tiles
                erode=False,  # Don't erode to avoid missing tissue
                prune=False   # Don't prune to avoid false negatives
            )
            
            print(f"Found {len(tile_coords)} tissue tiles for slide {slide_name}")
            
            # Create DataFrame with required format: x,y,slide,sample_id,target
            df_out = pd.DataFrame(tile_coords, columns=['x', 'y'])
            df_out['slide'] = slide_file  # Full slide path
            df_out['sample_id'] = row.iloc[0]  # Sample ID from first column
            if target_column:
                df_out['target'] = row.iloc[1]  # Target value from second column
            else:
                df_out['target'] = 'Unknown'  # Default if no target column
            
            # Categorize by resolution
            try:
                resolution = slide.properties[openslide.PROPERTY_NAME_OBJECTIVE_POWER]
                if resolution == '20':
                    out_20x.append(df_out)
                elif resolution == '40':
                    out_40x.append(df_out)
                else:
                    print(f"Slide {row.iloc[0]} has an unknown resolution: {resolution}")
            except KeyError:
                # Default to 40x if we can't determine
                print(f"Defaulting to 40x for slide {row.iloc[0]}")
                out_40x.append(df_out)
                
        except openslide.OpenSlideError:
            print(f"Slide {row.iloc[0]} not available.")
            missed_slides.append(row.iloc[0])
        except Exception as e:
            print(f"Error processing slide {row.iloc[0]}: {e}")
            missed_slides.append(row.iloc[0])
    
    # Save results with required column format: x,y,slide,sample_id,target
    if out_20x:
        out_20x = pd.concat(out_20x)
        # Ensure correct column order
        out_20x = out_20x[['x', 'y', 'slide', 'sample_id', 'target']]
        out_20x.to_csv(f'{output_path}/tile_coords_20x.csv', index=False)
        print(f"Saved {len(out_20x)} 20x tile coordinates")
    else:
        print("No 20x slides to process.")
        
    if out_40x:
        out_40x = pd.concat(out_40x)
        # Ensure correct column order
        out_40x = out_40x[['x', 'y', 'slide', 'sample_id', 'target']]
        out_40x.to_csv(f'{output_path}/tile_coords_40x.csv', index=False)
        print(f"Saved {len(out_40x)} 40x tile coordinates")
    else:
        print("No 40x slides to process.")

def main():
    parser = argparse.ArgumentParser(description="Process some CSV files.")
    parser.add_argument('--splits_csv', type=str, required=True, help='Path to the splits CSV file')
    parser.add_argument('--target', type=str, required=True, help='target column name CSV file')
    parser.add_argument('--slide_path', type=str, required=True, help='Prefix path for the slides')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the output CSV file')
    parser.add_argument('--tile_size', type=int, required=True, help='Size of tiles to extract')
    parser.add_argument('--config', type=str, help='Configuration file (optional)')
    
    args = parser.parse_args()
    
    # Load configuration if provided
    config = {}
    if args.config:
        config = load_config(args.config)
    
    # Use command line arguments
    manifest_path = args.splits_csv
    target_column = args.target
    slide_path = args.slide_path
    output_path = args.output_path
    tile_size = args.tile_size
    
    # Validate inputs
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")
    if not os.path.exists(slide_path):
        raise FileNotFoundError(f"Slide directory not found: {slide_path}")
    
    print(f"Processing manifest: {manifest_path}")
    print(f"Slide directory: {slide_path}")
    print(f"Output directory: {output_path}")
    print(f"Tile size: {tile_size}")
    print(f"Target column: {target_column}")
    
    # Process manifest using the correct implementation
    process_manifest(
        manifest_path, slide_path, output_path, tile_size, 
        overlap=1, min_cc_size=10, max_tiles=None, target_column=target_column
    )
    
    print("Tiling completed successfully")

if __name__ == "__main__":
    main() 