#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ComfyUI Blind Watermark - Utility Functions
"""

import torch
import numpy as np
from PIL import Image
import cv2


def tensor_to_cv2(tensor):
    """
    Convert ComfyUI tensor to OpenCV image
    
    Args:
        tensor: ComfyUI image tensor [B, H, W, C] in range [0, 1]
    
    Returns:
        OpenCV image in BGR format, uint8
    """
    # Take first image if batch
    if len(tensor.shape) == 4:
        tensor = tensor[0]
    
    # Convert to numpy
    img = tensor.cpu().numpy()
    
    # Convert range [0, 1] to [0, 255]
    img = (img * 255).astype(np.uint8)
    
    # Convert RGB to BGR for OpenCV
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    return img


def cv2_to_tensor(img):
    """
    Convert OpenCV image to ComfyUI tensor
    
    Args:
        img: OpenCV image in BGR format, uint8
    
    Returns:
        ComfyUI tensor [1, H, W, C] in range [0, 1]
    """
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Convert to float and normalize
    img = img.astype(np.float32) / 255.0
    
    # Convert to tensor with batch dimension
    tensor = torch.from_numpy(img).unsqueeze(0)
    
    return tensor


def tensor_batch_to_cv2_list(tensor):
    """
    Convert batch of ComfyUI tensors to list of OpenCV images
    
    Args:
        tensor: ComfyUI image tensor [B, H, W, C] in range [0, 1]
    
    Returns:
        List of OpenCV images in BGR format, uint8
    """
    batch_size = tensor.shape[0]
    images = []
    
    for i in range(batch_size):
        img = tensor[i].cpu().numpy()
        img = (img * 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        images.append(img)
    
    return images


def cv2_list_to_tensor_batch(images):
    """
    Convert list of OpenCV images to batch of ComfyUI tensors
    
    Args:
        images: List of OpenCV images in BGR format, uint8
    
    Returns:
        ComfyUI tensor [B, H, W, C] in range [0, 1]
    """
    tensors = []
    
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        tensors.append(torch.from_numpy(img))
    
    return torch.stack(tensors)


def calculate_d_params(strength):
    """
    Calculate d1 and d2 parameters based on strength
    
    Args:
        strength: Watermark strength multiplier (0.1 - 2.0)
    
    Returns:
        tuple: (d1, d2) parameters
    """
    # Default values: d1=36, d2=20
    d1 = int(36 * strength)
    d2 = int(20 * strength)
    
    # Ensure minimum values
    d1 = max(d1, 4)
    d2 = max(d2, 2)
    
    return d1, d2


def parse_processes_param(processes_str):
    """
    Parse processes parameter string
    
    Args:
        processes_str: String like "auto", "1", "4", "multiprocessing", etc.
    
    Returns:
        Appropriate processes parameter for WaterMark class
    """
    if processes_str == "auto":
        return None
    elif processes_str in ["multiprocessing", "multithreading"]:
        return processes_str
    else:
        try:
            return int(processes_str)
        except:
            return None


def format_watermark_info(wm_length=None, wm_width=None, wm_height=None, 
                         password_img=None, password_wm=None, strength=None):
    """
    Format watermark information as readable string
    
    Returns:
        Formatted string with watermark info
    """
    info_parts = ["=== Watermark Information ==="]
    
    if wm_length is not None:
        info_parts.append(f"Watermark Length: {wm_length} bits")
        info_parts.append(f"Estimated Text Length: ~{wm_length // 8} bytes")
    
    if wm_width is not None and wm_height is not None:
        info_parts.append(f"Watermark Size: {wm_width} x {wm_height}")
        info_parts.append(f"Total Pixels: {wm_width * wm_height}")
    
    if password_img is not None:
        info_parts.append(f"Image Password: {password_img}")
    
    if password_wm is not None:
        info_parts.append(f"Watermark Password: {password_wm}")
    
    if strength is not None:
        d1, d2 = calculate_d_params(strength)
        info_parts.append(f"Strength: {strength}")
        info_parts.append(f"  -> d1={d1}, d2={d2}")
        
        if strength < 0.5:
            info_parts.append("  -> Quality: Low robustness, high image quality")
        elif strength > 1.5:
            info_parts.append("  -> Quality: High robustness, lower image quality")
        else:
            info_parts.append("  -> Quality: Balanced")
    
    return "\n".join(info_parts)

