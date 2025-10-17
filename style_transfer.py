"""
Real-Time Neural Style Transfer Studio
Built with PyTorch and Gradio

Features:
- Fast Style Transfer using pre-trained models
- Real-time webcam processing
- Multiple artistic styles
- Style intensity control
- High-quality image output
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import vgg19
from PIL import Image
import numpy as np
import gradio as gr
import cv2
from pathlib import Path
import urllib.request
import os
import zipfile
import shutil

# Limit CPU thread usage
torch.set_num_threads(2)

# Disable gradients for inference
torch.set_grad_enabled(False)


# ============================================================================
# STYLE TRANSFER MODEL (Fast Neural Style)
# ============================================================================

class TransformerNet(nn.Module):
    """
    Fast Style Transfer Network
    Based on Johnson et al. "Perceptual Losses for Real-Time Style Transfer"
    """
    def __init__(self):
        super(TransformerNet, self).__init__()
        
        # Initial convolution layers
        self.conv1 = ConvLayer(3, 32, kernel_size=9, stride=1)
        self.in1 = nn.InstanceNorm2d(32, affine=True)
        self.conv2 = ConvLayer(32, 64, kernel_size=3, stride=2)
        self.in2 = nn.InstanceNorm2d(64, affine=True)
        self.conv3 = ConvLayer(64, 128, kernel_size=3, stride=2)
        self.in3 = nn.InstanceNorm2d(128, affine=True)
        
        # Residual layers
        self.res1 = ResidualBlock(128)
        self.res2 = ResidualBlock(128)
        self.res3 = ResidualBlock(128)
        self.res4 = ResidualBlock(128)
        self.res5 = ResidualBlock(128)
        
        # Upsampling layers
        self.deconv1 = UpsampleConvLayer(128, 64, kernel_size=3, stride=1, upsample=2)
        self.in4 = nn.InstanceNorm2d(64, affine=True)
        self.deconv2 = UpsampleConvLayer(64, 32, kernel_size=3, stride=1, upsample=2)
        self.in5 = nn.InstanceNorm2d(32, affine=True)
        self.deconv3 = ConvLayer(32, 3, kernel_size=9, stride=1)
        
        # Non-linearity
        self.relu = nn.ReLU()

    def forward(self, x):
        y = self.relu(self.in1(self.conv1(x)))
        y = self.relu(self.in2(self.conv2(y)))
        y = self.relu(self.in3(self.conv3(y)))
        y = self.res1(y)
        y = self.res2(y)
        y = self.res3(y)
        y = self.res4(y)
        y = self.res5(y)
        y = self.relu(self.in4(self.deconv1(y)))
        y = self.relu(self.in5(self.deconv2(y)))
        y = self.deconv3(y)
        return y


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in1 = nn.InstanceNorm2d(channels, affine=True)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in2 = nn.InstanceNorm2d(channels, affine=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        out = out + residual
        return out


class UpsampleConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        if upsample:
            self.upsample_layer = nn.Upsample(scale_factor=upsample, mode='nearest')
        padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        if self.upsample:
            x = self.upsample_layer(x)
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out


# ============================================================================
# STYLE TRANSFER ENGINE
# ============================================================================

class StyleTransferEngine:
    """
    Handles model loading, inference, and style blending
    """
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.models = {}
        
        # Dropbox link for all pre-trained models
        self.models_zip_url = 'https://www.dropbox.com/s/lrvwfehqdcxoza8/saved_models.zip?dl=1'
        self.available_models = ['mosaic', 'candy', 'rain_princess', 'udnie']
        
        # Create models directory
        self.models_dir = Path('style_models')
        self.models_dir.mkdir(exist_ok=True)
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.mul(255))
        ])
    
    def download_all_models(self):
        """Download and extract all pre-trained models from Dropbox"""
        # Check if any model is missing
        missing_models = [m for m in self.available_models 
                         if not (self.models_dir / f'{m}.pth').exists()]
        
        if not missing_models:
            return True
        
        print(f"Downloading models from Dropbox...")
        zip_path = self.models_dir / 'saved_models.zip'
        
        try:
            # Download the zip file
            urllib.request.urlretrieve(self.models_zip_url, zip_path)
            print("✓ Downloaded models archive")
            
            # Extract the zip file
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # Extract to temporary location
                temp_extract = self.models_dir / 'temp'
                temp_extract.mkdir(exist_ok=True)
                zip_ref.extractall(temp_extract)
                
                # Move .pth files to models directory
                for model_file in temp_extract.glob('**/*.pth'):
                    target = self.models_dir / model_file.name
                    model_file.rename(target)
                    print(f"✓ Extracted {model_file.name}")
                
                # Clean up
                shutil.rmtree(temp_extract, ignore_errors=True)
            
            # Remove zip file
            zip_path.unlink()
            print("✓ All models ready")
            return True
            
        except Exception as e:
            print(f"✗ Error downloading models: {e}")
            if zip_path.exists():
                zip_path.unlink()
            return False
        
    def load_model(self, style_name):
        """Load or retrieve cached model"""
        if style_name in self.models:
            return self.models[style_name]
        
        model_path = self.models_dir / f'{style_name}.pth'
        
        # Download models if not present
        if not model_path.exists():
            print(f"Model {style_name} not found, downloading...")
            if not self.download_all_models():
                return None
        
        if not model_path.exists():
            print(f"✗ Could not find {style_name} model")
            return None
        
        # Load model
        try:
            model = TransformerNet()
            state_dict = torch.load(model_path, map_location=self.device)
            
            # Remove 'module.' prefix if present (from DataParallel)
            new_state_dict = {}
            for k, v in state_dict.items():
                new_key = k.replace('module.', '') if k.startswith('module.') else k
                new_state_dict[new_key] = v
            
            # Remove running stats from InstanceNorm2d (not needed for inference)
            keys_to_remove = [k for k in new_state_dict.keys() if 'running_mean' in k or 'running_var' in k]
            for k in keys_to_remove:
                del new_state_dict[k]
            
            model.load_state_dict(new_state_dict, strict=False)
            model.to(self.device)
            model.eval()
            
            self.models[style_name] = model
            print(f"✓ Loaded {style_name} model")
            return model
        except Exception as e:
            print(f"✗ Error loading {style_name}: {e}")
            return None
    
    def stylize(self, image, style_name, intensity=1.0):
        """
        Apply style transfer to image
        
        Args:
            image: PIL Image or numpy array
            style_name: Name of the style to apply
            intensity: Blend factor (0=original, 1=full style)
        
        Returns:
            Styled PIL Image
        """
        # Convert to PIL if numpy
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        original_image = image.copy()
        
        # Load model
        model = self.load_model(style_name)
        if model is None:
            return original_image
        
        # Preprocess
        img_tensor = self.transform(image).unsqueeze(0).to(self.device) # type: ignore
        
        # Inference
        with torch.no_grad():
            output = model(img_tensor)
        
        # Postprocess
        output = output[0].cpu().clamp(0, 255).numpy()
        output = output.transpose(1, 2, 0).astype('uint8')
        styled_image = Image.fromarray(output)
        
        # Blend with original based on intensity
        if intensity < 1.0:
            styled_image = Image.blend(original_image, styled_image, intensity)
        
        return styled_image


# ============================================================================
# GRADIO INTERFACE
# ============================================================================

class StyleTransferApp:
    """
    Gradio application for Style Transfer Studio
    """
    def __init__(self):
        self.engine = StyleTransferEngine()
        self.available_styles = self.engine.available_models
        
        # Download all models at startup
        print("Initializing models...")
        self.engine.download_all_models()
    
    def process_image(self, image, style, intensity):
        """Process single image"""
        if image is None:
            return None
        
        try:
            styled = self.engine.stylize(image, style, intensity / 100.0)
            return np.array(styled)
        except Exception as e:
            print(f"Error: {e}")
            return image
    
    def process_video_frame(self, frame, style, intensity):
        """Process video frame (for webcam)"""
        return self.process_image(frame, style, intensity)
    
    def create_interface(self):
        """Create Gradio interface"""
        
        with gr.Blocks(theme=gr.themes.Soft(), title="Style Transfer Studio") as demo: # type: ignore
            gr.Markdown("""
            # 🎨 Neural Style Transfer Studio
            Transform your images into artistic masterpieces using deep learning!
            
            **Quick Start:**
            1. Upload an image or use your webcam
            2. Choose an artistic style
            3. Adjust the intensity slider
            4. Download your styled image
            """)
            
            with gr.Tabs():
                # IMAGE TAB
                with gr.Tab("📸 Image Style Transfer"):
                    with gr.Row():
                        with gr.Column():
                            image_input = gr.Image(label="Upload Image", type="numpy")
                            style_dropdown = gr.Dropdown(
                                choices=self.available_styles,
                                value=self.available_styles[0],
                                label="🎨 Select Style"
                            )
                            intensity_slider = gr.Slider(
                                minimum=0,
                                maximum=100,
                                value=100,
                                step=5,
                                label="Style Intensity (%)"
                            )
                            process_btn = gr.Button("✨ Apply Style", variant="primary")
                        
                        with gr.Column():
                            image_output = gr.Image(label="Styled Image")
                    
                    process_btn.click(
                        fn=self.process_image,
                        inputs=[image_input, style_dropdown, intensity_slider],
                        outputs=image_output
                    )
                    
                    # Example images
                    gr.Examples(
                        examples=[
                            ["examples/example1.jpg", "mosaic", 100],
                            ["examples/example2.jpg", "candy", 100],
                        ] if Path("examples").exists() else [],
                        inputs=[image_input, style_dropdown, intensity_slider],
                        outputs=image_output,
                        fn=self.process_image,
                        cache_examples=False
                    )
                
                # WEBCAM TAB
                with gr.Tab("📹 Real-Time Webcam"):
                    gr.Markdown("⚠️ Note: Processing may be slower on CPU. GPU recommended for smooth real-time performance.")
                    
                    with gr.Row():
                        with gr.Column():
                            webcam_input = gr.Image(sources=["webcam"], streaming=True, label="Webcam", type="numpy") # type: ignore
                            webcam_style = gr.Dropdown(
                                choices=self.available_styles,
                                value=self.available_styles[0],
                                label="🎨 Select Style"
                            )
                            webcam_intensity = gr.Slider(
                                minimum=0,
                                maximum=100,
                                value=80,
                                step=10,
                                label="Style Intensity (%)"
                            )
                        
                        with gr.Column():
                            webcam_output = gr.Image(label="Styled Output", streaming=True)
                    
                    webcam_input.stream(
                        fn=self.process_video_frame,
                        inputs=[webcam_input, webcam_style, webcam_intensity],
                        outputs=webcam_output
                    )
            
            # Info section
            with gr.Accordion("ℹ️ About & Tips", open=False):
                gr.Markdown("""
                ### Available Styles
                - **Mosaic**: Abstract geometric patterns
                - **Candy**: Vibrant, colorful pop art
                - **Rain Princess**: Impressionist painting style
                - **Udnie**: Abstract expressionism
                
                ### Performance Tips
                - Use GPU for real-time webcam processing
                - Lower intensity for subtle effects
                - Recommended image size: 512-1024px
                
                ### Technical Details
                - Model: Fast Neural Style Transfer (Johnson et al.)
                - Framework: PyTorch
                - Pre-trained on MS-COCO dataset
                
                **Note:** First run will download models (~7MB each)
                """)
        
        return demo
    
    def launch(self):
        """Launch the application"""
        demo = self.create_interface()
        demo.launch(
            share=True,
            server_name="0.0.0.0",
            server_port=7860,
            show_error=True
        )


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("🎨 STYLE TRANSFER STUDIO")
    print("=" * 60)
    print(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    print("=" * 60)
    
    app = StyleTransferApp()
    app.launch()