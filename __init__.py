#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ComfyUI Blind Watermark Plugin
Advanced Blind Watermark Embedding and Extraction for ComfyUI

GitHub: https://github.com/guofei9987/blind_watermark
"""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

# Export for ComfyUI
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

# Version info
__version__ = "1.0.0"
__author__ = "ComfyUI Blind Watermark Team"

# Print loading message
print("\n" + "=" * 60)
print("🔐 ComfyUI Blind Watermark Plugin Loaded")
print("=" * 60)
print(f"Version: {__version__}")
print(f"Nodes loaded: {len(NODE_CLASS_MAPPINGS)}")
print("\nAvailable Nodes:")
for node_name, display_name in NODE_DISPLAY_NAME_MAPPINGS.items():
    print(f"  • {display_name}")
print("=" * 60 + "\n")
