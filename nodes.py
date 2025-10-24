
"""
ComfyUI Blind Watermark - Node Implementations
Advanced Mode with Full Control
"""

import torch
import numpy as np
import cv2
import qrcode
from qrcode.constants import ERROR_CORRECT_L, ERROR_CORRECT_M, ERROR_CORRECT_Q, ERROR_CORRECT_H
from PIL import Image as PILImage

# Try pyzbar import (optional for QR decoding)
try:
    from pyzbar.pyzbar import decode as pyzbar_decode
    PYZBAR_AVAILABLE = True
except ImportError:
    PYZBAR_AVAILABLE = False
    print("Warning: pyzbar not installed. QR code decoding will not be available.")

# Try relative import first (for ComfyUI), fall back to absolute (for testing)
try:
    from .blind_watermark import WaterMark
    from .utils import (
        tensor_to_cv2, cv2_to_tensor, 
        calculate_d_params, parse_processes_param, format_watermark_info
    )
except ImportError:
    from blind_watermark import WaterMark
    from utils import (
        tensor_to_cv2, cv2_to_tensor, 
        calculate_d_params, parse_processes_param, format_watermark_info
    )


# ===================================================================
#  Text Watermark Embed Node
# ===================================================================

class WatermarkEmbedText:
    """
    Embed text watermark into image
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "watermark_text": ("STRING", {
                    "default": "Copyright 2024",
                    "multiline": True
                }),
                "password_img": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 999999,
                }),
                "password_wm": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 999999,
                }),
                "strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 3.0,
                    "step": 0.1,
                }),
                "processes": ([
                    "auto", "1", "2", "4", "8",
                    "multiprocessing", "multithreading"
                ], {"default": "auto"}),
            },
            "optional": {
                "d1_override": ("INT", {"default": 0, "min": 0, "max": 200, "tooltip": "Manual d1 parameter (0=auto)"}),
                "d2_override": ("INT", {"default": 0, "min": 0, "max": 100, "tooltip": "Manual d2 parameter (0=auto)"}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "INT", "STRING")
    RETURN_NAMES = ("watermarked_image", "wm_length", "info")
    FUNCTION = "embed_watermark"
    CATEGORY = "BlindWatermark/Embed"
    
    def embed_watermark(self, image, watermark_text, password_img, password_wm,
                       strength, processes, d1_override=0, d2_override=0):
        """Embed text watermark into image"""
        
        # Calculate d1, d2
        if d1_override > 0 and d2_override > 0:
            d1, d2 = d1_override, d2_override
        else:
            d1, d2 = calculate_d_params(strength)
        processes_param = parse_processes_param(processes)
        
        # Convert tensor to cv2 image
        cv2_img = tensor_to_cv2(image)
        
        # Create WaterMark instance
        bwm = WaterMark(
            password_wm=password_wm,
            password_img=password_img,
            processes=processes_param
        )
        
        # Manually set d1, d2
        bwm.bwm_core.d1 = d1
        bwm.bwm_core.d2 = d2
        
        # Read image and watermark
        bwm.read_img(img=cv2_img)
        bwm.read_wm(watermark_text, mode='str')
        
        # Embed watermark
        output_img = bwm.embed()
        
        # Convert back to tensor
        output_tensor = cv2_to_tensor(output_img)
        
        # Get watermark length
        wm_length = bwm.wm_size
        
        # Generate info
        info = format_watermark_info(
            wm_length=wm_length,
            password_img=password_img,
            password_wm=password_wm,
            strength=strength
        )
        
        return (output_tensor, wm_length, info)


# ===================================================================
#  Text Watermark Extract Node
# ===================================================================

class WatermarkExtractText:
    """
    Extract text watermark from image
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "wm_length": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 999999,
                    "tooltip": "Watermark length (get from embed node)"
                }),
                "password_img": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 999999,
                }),
                "password_wm": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 999999,
                }),
                "processes": ([
                    "auto", "1", "2", "4", "8",
                    "multiprocessing", "multithreading"
                ], {"default": "auto"}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("extracted_text",)
    FUNCTION = "extract_watermark"
    CATEGORY = "BlindWatermark/Extract"
    
    def extract_watermark(self, image, wm_length, password_img, password_wm, processes):
        """Extract text watermark from image"""
        
        processes_param = parse_processes_param(processes)
        
        # Convert tensor to cv2 image
        cv2_img = tensor_to_cv2(image)
        
        # Validate image size
        height, width = cv2_img.shape[:2]
        min_size = max(int(np.sqrt(wm_length) * 4), 128)
        
        if height < min_size or width < min_size:
            error_msg = f"Image too small for watermark extraction. Minimum size: {min_size}x{min_size}, got: {width}x{height}"
            print(f"[Blind Watermark Error] {error_msg}")
            return (f"ERROR: {error_msg}",)
        
        # Create WaterMark instance
        bwm = WaterMark(
            password_wm=password_wm,
            password_img=password_img,
            processes=processes_param
        )
        
        try:
            # Extract watermark
            extracted_text = bwm.extract(
                embed_img=cv2_img,
                wm_shape=wm_length,
                mode='str'
            )
            
            return (extracted_text,)
        except ValueError as e:
            error_msg = f"Failed to extract watermark: {str(e)}. Check if image contains watermark with these parameters."
            print(f"[Blind Watermark Error] {error_msg}")
            return (f"ERROR: {error_msg}",)
        except Exception as e:
            error_msg = f"Extraction failed: {str(e)}"
            print(f"[Blind Watermark Error] {error_msg}")
            return (f"ERROR: {error_msg}",)


# ===================================================================
#  Image Watermark Embed Node
# ===================================================================

class WatermarkEmbedImage:
    """
    Embed image watermark into image
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "watermark_image": ("IMAGE",),
                "password_img": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 999999,
                }),
                "password_wm": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 999999,
                }),
                "strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 3.0,
                    "step": 0.1,
                }),
                "processes": ([
                    "auto", "1", "2", "4", "8",
                    "multiprocessing", "multithreading"
                ], {"default": "auto"}),
            },
            "optional": {
                "d1_override": ("INT", {"default": 0, "min": 0, "max": 200, "tooltip": "Manual d1 parameter (0=auto)"}),
                "d2_override": ("INT", {"default": 0, "min": 0, "max": 100, "tooltip": "Manual d2 parameter (0=auto)"}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("watermarked_image", "info")
    FUNCTION = "embed_watermark"
    CATEGORY = "BlindWatermark/Embed"
    
    def embed_watermark(self, image, watermark_image, password_img, password_wm,
                       strength, processes, d1_override=0, d2_override=0):
        """Embed image watermark into image"""
        
        # Calculate d1, d2
        if d1_override > 0 and d2_override > 0:
            d1, d2 = d1_override, d2_override
        else:
            d1, d2 = calculate_d_params(strength)
        processes_param = parse_processes_param(processes)
        
        # Convert tensors to cv2 images
        cv2_img = tensor_to_cv2(image)
        cv2_wm = tensor_to_cv2(watermark_image)
        
        # Convert watermark to grayscale
        import cv2
        cv2_wm_gray = cv2.cvtColor(cv2_wm, cv2.COLOR_BGR2GRAY)
        
        # Get watermark dimensions
        wm_height, wm_width = cv2_wm_gray.shape
        
        # Create WaterMark instance
        bwm = WaterMark(
            password_wm=password_wm,
            password_img=password_img,
            processes=processes_param
        )
        
        # Manually set d1, d2
        bwm.bwm_core.d1 = d1
        bwm.bwm_core.d2 = d2
        
        # Read image
        bwm.read_img(img=cv2_img)
        
        # Convert watermark to bit array
        wm_bit = cv2_wm_gray.flatten() > 128
        bwm.wm_bit = wm_bit
        bwm.wm_size = wm_bit.size
        
        # Shuffle watermark
        np.random.RandomState(password_wm).shuffle(bwm.wm_bit)
        bwm.bwm_core.read_wm(bwm.wm_bit)
        
        # Embed watermark
        output_img = bwm.embed()
        
        # Convert back to tensor
        output_tensor = cv2_to_tensor(output_img)
        
        # Generate info
        info = format_watermark_info(
            wm_width=wm_width,
            wm_height=wm_height,
            password_img=password_img,
            password_wm=password_wm,
            strength=strength
        )
        
        return (output_tensor, info)


# ===================================================================
#  Image Watermark Extract Node
# ===================================================================

class WatermarkExtractImage:
    """
    Extract image watermark from image
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "wm_width": ("INT", {
                    "default": 128,
                    "min": 1,
                    "max": 2048,
                }),
                "wm_height": ("INT", {
                    "default": 128,
                    "min": 1,
                    "max": 2048,
                }),
                "password_img": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 999999,
                }),
                "password_wm": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 999999,
                }),
                "processes": ([
                    "auto", "1", "2", "4", "8",
                    "multiprocessing", "multithreading"
                ], {"default": "auto"}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("extracted_watermark",)
    FUNCTION = "extract_watermark"
    CATEGORY = "BlindWatermark/Extract"
    
    def extract_watermark(self, image, wm_width, wm_height, password_img, 
                         password_wm, processes):
        """Extract image watermark from image"""
        
        processes_param = parse_processes_param(processes)
        
        # Convert tensor to cv2 image
        cv2_img = tensor_to_cv2(image)
        
        # Validate image size
        height, width = cv2_img.shape[:2]
        wm_size = wm_width * wm_height
        min_size = max(int(np.sqrt(wm_size) * 4), 128)
        
        if height < min_size or width < min_size:
            error_msg = f"Image too small for watermark extraction. Minimum size: {min_size}x{min_size}, got: {width}x{height}"
            print(f"[Blind Watermark Error] {error_msg}")
            # Return a black image with error message
            black_img = np.zeros((wm_height, wm_width, 3), dtype=np.uint8)
            return (cv2_to_tensor(black_img),)
        
        # Create WaterMark instance
        bwm = WaterMark(
            password_wm=password_wm,
            password_img=password_img,
            processes=processes_param
        )
        
        # Extract watermark
        import tempfile
        import os
        
        # Create temporary file for watermark extraction
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            wm_extract = bwm.extract(
                embed_img=cv2_img,
                wm_shape=(wm_height, wm_width),
                out_wm_name=tmp_path,
                mode='img'
            )
            
            # Read the extracted watermark
            import cv2
            wm_extract_img = cv2.imread(tmp_path, cv2.IMREAD_GRAYSCALE)
            
            if wm_extract_img is None:
                raise ValueError("Failed to read extracted watermark")
            
            # Convert to RGB image
            wm_extract_rgb = cv2.cvtColor(wm_extract_img, cv2.COLOR_GRAY2RGB)
        except ValueError as e:
            error_msg = f"Failed to extract watermark: {str(e)}. Check if image contains watermark with these parameters."
            print(f"[Blind Watermark Error] {error_msg}")
            # Return a black image
            black_img = np.zeros((wm_height, wm_width, 3), dtype=np.uint8)
            wm_extract_rgb = black_img
        except Exception as e:
            error_msg = f"Extraction failed: {str(e)}"
            print(f"[Blind Watermark Error] {error_msg}")
            # Return a black image
            black_img = np.zeros((wm_height, wm_width, 3), dtype=np.uint8)
            wm_extract_rgb = black_img
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        
        # Convert to tensor
        output_tensor = cv2_to_tensor(wm_extract_rgb)
        
        return (output_tensor,)



# ===================================================================
#  Node Registration Mapping
# ===================================================================

# ===================================================================
#  QR Code Generator Node
# ===================================================================

class WatermarkQRCodeGenerator:
    """
    Generate QR Code image from text/URL
    Can be used as watermark or standalone
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {
                    "default": "https://github.com/guofei9987/blind_watermark",
                    "multiline": True,
                    "tooltip": "Text or URL to encode in QR code"
                }),
                "qr_size": ("INT", {
                    "default": 128,
                    "min": 32,
                    "max": 1024,
                    "step": 8,
                    "tooltip": "QR code image size (pixels)"
                }),
                "error_correction": ([
                    "L (7%)", "M (15%)", "Q (25%)", "H (30%)"
                ], {
                    "default": "M (15%)",
                    "tooltip": "Error correction level - higher allows more damage tolerance"
                }),
                "border": ("INT", {
                    "default": 2,
                    "min": 0,
                    "max": 10,
                    "tooltip": "Border size in modules"
                }),
                "invert": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Invert colors (white on black)"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("qr_image", "info")
    FUNCTION = "generate_qr"
    CATEGORY = "BlindWatermark/QRCode"
    
    def generate_qr(self, text, qr_size, error_correction, border, invert):
        """Generate QR code image"""
        
        # Map error correction level
        error_map = {
            "L (7%)": ERROR_CORRECT_L,
            "M (15%)": ERROR_CORRECT_M,
            "Q (25%)": ERROR_CORRECT_Q,
            "H (30%)": ERROR_CORRECT_H,
        }
        error_level = error_map[error_correction]
        
        # Create QR code
        qr = qrcode.QRCode(
            version=None,  # Auto-detect version based on data
            error_correction=error_level,
            box_size=10,
            border=border,
        )
        qr.add_data(text)
        qr.make(fit=True)
        
        # Generate PIL image
        fill_color = "white" if invert else "black"
        back_color = "black" if invert else "white"
        pil_img = qr.make_image(fill_color=fill_color, back_color=back_color)
        
        # Resize to specified size
        pil_img = pil_img.resize((qr_size, qr_size), PILImage.Resampling.NEAREST)
        
        # Convert to numpy array
        qr_array = np.array(pil_img.convert('L'))  # Convert to grayscale
        
        # Convert to ComfyUI tensor format [B, H, W, C]
        qr_tensor = torch.from_numpy(qr_array).float() / 255.0
        qr_tensor = qr_tensor.unsqueeze(0).unsqueeze(-1).repeat(1, 1, 1, 3)
        
        # Info
        info = f"✅ QR Code Generated\n"
        info += f"Size: {qr_size}x{qr_size}\n"
        info += f"Text Length: {len(text)} chars\n"
        info += f"Error Correction: {error_correction}\n"
        info += f"Version: {qr.version}"
        
        return (qr_tensor, info)


# ===================================================================
#  QR Code Decoder Node
# ===================================================================

class WatermarkQRCodeDecoder:
    """
    Decode QR Code from image
    Useful for extracting watermark QR codes
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "threshold": ("INT", {
                    "default": 128,
                    "min": 0,
                    "max": 255,
                    "tooltip": "Binarization threshold (0=auto)"
                }),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("decoded_text", "info")
    FUNCTION = "decode_qr"
    CATEGORY = "BlindWatermark/QRCode"
    
    def decode_qr(self, image, threshold):
        """Decode QR code from image"""
        
        if not PYZBAR_AVAILABLE:
            return ("", "❌ Error: pyzbar not installed\nRun: pip install pyzbar")
        
        # Convert tensor to cv2 image
        cv2_img = tensor_to_cv2(image)
        
        # Convert to grayscale
        if len(cv2_img.shape) == 3:
            gray = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = cv2_img
        
        # Apply threshold if not auto
        if threshold > 0:
            _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        else:
            # Auto threshold using Otsu's method
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Try to decode QR code
        decoded_objects = pyzbar_decode(binary)
        
        if not decoded_objects:
            # Try with inverted image
            decoded_objects = pyzbar_decode(255 - binary)
        
        if decoded_objects:
            # Get the first decoded object
            obj = decoded_objects[0]
            decoded_text = obj.data.decode('utf-8', errors='replace')
            
            info = f"✅ QR Code Decoded Successfully\n"
            info += f"Type: {obj.type}\n"
            info += f"Text Length: {len(decoded_text)} chars\n"
            info += f"Quality: {obj.quality if hasattr(obj, 'quality') else 'N/A'}"
            
            return (decoded_text, info)
        else:
            info = "❌ No QR Code Found\n"
            info += "Tips:\n"
            info += "• Adjust threshold parameter\n"
            info += "• Ensure QR code is clear and centered\n"
            info += "• Check if image is watermarked correctly"
            
            return ("", info)


NODE_CLASS_MAPPINGS = {
    "WatermarkEmbedText": WatermarkEmbedText,
    "WatermarkExtractText": WatermarkExtractText,
    "WatermarkEmbedImage": WatermarkEmbedImage,
    "WatermarkExtractImage": WatermarkExtractImage,
    "WatermarkQRCodeGenerator": WatermarkQRCodeGenerator,
    "WatermarkQRCodeDecoder": WatermarkQRCodeDecoder,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WatermarkEmbedText": "📝 Embed Text Watermark",
    "WatermarkExtractText": "📖 Extract Text Watermark",
    "WatermarkEmbedImage": "🖼️ Embed Image Watermark",
    "WatermarkExtractImage": "🔍 Extract Image Watermark",
    "WatermarkQRCodeGenerator": "📲 Generate QR Code",
    "WatermarkQRCodeDecoder": "🔍 Decode QR Code",
}

