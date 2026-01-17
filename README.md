# 🏠 Flooring Visualizer

AI-powered tool to visualize different flooring textures in room photos. Simply upload a room image and a floor texture - the AI automatically detects the floor and applies the new texture with realistic lighting.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Gradio](https://img.shields.io/badge/Gradio-4.0+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## ✨ Features

- **Automatic Floor Detection**: Uses SegFormer (nvidia/segformer-b5-finetuned-ade-640-640) trained on ADE20K
- **Realistic Blending**: Preserves original room lighting and shadows
- **Perspective Matching**: Applies texture with correct perspective transform
- **Modern UI**: Beautiful dark theme with Gradio interface
- **No Manual Input**: Fully automatic - no clicking or selecting required

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- CUDA GPU (optional, but recommended for faster inference)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/FlooringVisualizer.git
   cd FlooringVisualizer
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   # Using conda
   conda create -n flooring_env python=3.10
   conda activate flooring_env
   
   # Or using venv
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # venv\Scripts\activate   # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   python app.py
   ```

5. **Open in browser**
   Navigate to `http://localhost:7861`

## 📖 Usage

1. **Upload Room Photo**: Select an image of the room you want to modify
2. **Upload Texture**: Select a flooring texture image (wood, tile, carpet, etc.)
3. **Adjust Blend**: Optionally adjust the blend strength for lighting preservation
4. **Generate**: Click "Generate Visualization" to see the result

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| **AI Model** | [SegFormer B5](https://huggingface.co/nvidia/segformer-b5-finetuned-ade-640-640) (ADE20K) |
| **Deep Learning** | PyTorch |
| **Image Processing** | OpenCV, PIL |
| **Web Interface** | Gradio |
| **Model Hub** | Hugging Face Transformers |

## 📁 Project Structure

```
FlooringVisualizer/
├── app.py              # Gradio web interface
├── core.py             # Core AI and image processing logic
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

## 🔧 Configuration

### Blend Strength
- **Higher values (0.7-1.0)**: Preserves more original lighting/shadows
- **Lower values (0.0-0.3)**: Shows more of the texture's original colors

### Server Settings
Edit `app.py` to change:
- `server_port`: Default is 7861
- `share`: Set to `True` to create a public URL

## 🤖 How It Works

1. **Floor Detection**: SegFormer segments the image and identifies floor pixels (ADE20K class 3)
2. **Perspective Transform**: The texture is warped to match the floor's perspective
3. **Lighting Blend**: LAB color space blending preserves original shadows and highlights
4. **Edge Feathering**: Smooth transitions at floor boundaries

## 📋 Requirements

```
torch
torchvision
opencv-python
numpy<2
gradio
Pillow
transformers
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [NVIDIA SegFormer](https://huggingface.co/nvidia/segformer-b5-finetuned-ade-640-640) for the semantic segmentation model
- [Hugging Face](https://huggingface.co/) for model hosting
- [Gradio](https://gradio.app/) for the web interface framework
