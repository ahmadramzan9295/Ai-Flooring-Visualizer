"""
app.py - Flooring Visualizer Gradio Interface

A web interface for visualizing different flooring textures in room photos.
Uses SegFormer (ADE20K) for automatic floor detection.
"""

import gradio as gr
import numpy as np
from PIL import Image
from typing import Tuple

from core import load_model, process_room

# Global model components (loaded once)
model = None
processor = None
device = None


def initialize_model():
    """Load the SegFormer model on startup."""
    global model, processor, device
    
    if model is None:
        model, processor, device = load_model()
    return model, processor, device


def visualize_floor(
    room_image: np.ndarray,
    texture_image: np.ndarray,
    blend_strength: float = 0.7
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Process room image and replace floor with texture.

    Args:
        room_image: Uploaded room image.
        texture_image: Uploaded texture image.
        blend_strength: How much original lighting to preserve.

    Returns:
        Tuple of (result_image, mask_visualization).
    """
    global model, processor, device
    if model is None:
        initialize_model()

    # Convert PIL to numpy if needed
    if isinstance(room_image, Image.Image):
        room_image = np.array(room_image)
    if isinstance(texture_image, Image.Image):
        texture_image = np.array(texture_image)

    if room_image is None or texture_image is None:
        return None, None

    # Process - no clicking needed, automatic detection!
    result, mask = process_room(
        room_image,
        texture_image,
        model,
        processor,
        device,
        blend_strength=blend_strength
    )

    # Create mask visualization
    mask_vis = np.zeros_like(room_image)
    mask_vis[:, :, 1] = mask * 255  # Green overlay
    mask_overlay = (room_image * 0.5 + mask_vis * 0.5).astype(np.uint8)

    return result, mask_overlay


# -----------------------------------------------------------------------------
# Custom CSS & Styling
# -----------------------------------------------------------------------------

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');

body {
    font-family: 'Outfit', sans-serif !important;
    background-color: #0f172a;
}

.gradio-container {
    background: radial-gradient(circle at top left, #1e293b, #0f172a) !important;
}

/* Header Styling */
.header-container {
    text-align: center;
    padding: 2rem 0;
    margin-bottom: 2rem;
    color: white;
}

.header-title {
    font-size: 3rem;
    font-weight: 700;
    background: linear-gradient(135deg, #60a5fa, #c084fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.5rem;
}

.header-subtitle {
    font-size: 1.2rem;
    color: #94a3b8;
    font-weight: 300;
}

/* Card/Container Styling */
.group-container {
    background-color: #1e293b !important;
    border: 1px solid #334155 !important;
    border-radius: 1rem !important;
    padding: 1.5rem !important;
    box-shadow: 0 10px 30px rgba(0,0,0,0.3) !important;
}

/* Button Styling */
#viz-btn {
    background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%) !important;
    border: none !important;
    color: white !important;
    font-weight: 600;
    font-size: 1.1rem;
    padding: 0.8rem 1.5rem;
    border-radius: 0.5rem;
    transition: transform 0.2s, box-shadow 0.2s;
}

#viz-btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 5px 15px rgba(139, 92, 246, 0.4);
}

/* Image Label Styling */
.block-label {
    background: transparent !important;
    color: #e2e8f0 !important;
    font-weight: 600;
    font-size: 0.9rem;
    margin-bottom: 0.5rem;
}
"""

def create_interface():
    """Create and return the Gradio interface with custom styling."""
    
    # Define a custom theme foundation to tweak
    theme = gr.themes.Soft(
        primary_hue="indigo",
        secondary_hue="slate",
        neutral_hue="slate",
        font=["Outfit", "sans-serif"]
    ).set(
        body_background_fill="#0f172a",
        block_background_fill="#1e293b",
        block_border_width="0px",
        block_title_text_color="#e2e8f0",
        block_label_text_color="#94a3b8",
        input_background_fill="#334155",
    )

    with gr.Blocks(title="Flooring Visualizer", theme=theme, css=CUSTOM_CSS) as demo:
        
        # Header
        with gr.Row(elem_classes=["header-row"]):
            gr.HTML(
                """
                <div class="header-container">
                    <div class="header-title">Flooring Visualizer</div>
                    <div class="header-subtitle">Reimagine your space with AI-powered automatic floor detection</div>
                </div>
                """
            )

        with gr.Row():
            # LEFT COLUMN: Inputs
            with gr.Column(scale=4, variant="panel", elem_classes=["group-container"]):
                gr.Markdown("### 🛠️ Configuration")
                
                with gr.Tabs():
                    with gr.TabItem("Room Photo"):
                        room_input = gr.Image(
                            label="Upload Room Photo",
                            type="numpy",
                            height=350,
                            elem_id="room-img"
                        )
                    
                    with gr.TabItem("Texture"):
                        texture_input = gr.Image(
                            label="New Flooring Texture",
                            type="numpy",
                            height=350,
                            elem_id="texture-img"
                        )
                
                with gr.Accordion("🎨 Advanced Blending", open=False):
                    blend_slider = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.7,
                        step=0.05,
                        label="Texture Blend Strength",
                        info="Adjust to balance between texture visibility and original shadows."
                    )
                
                gr.HTML("<br>")
                
                process_btn = gr.Button(
                    "✨ Generate Visualization",
                    variant="primary",
                    elem_id="viz-btn"
                )
                
                gr.Markdown(
                    """
                    <div style="text-align: center; color: #64748b; margin-top: 1rem; font-size: 0.85rem;">
                    💡 <b>Tip:</b> Just upload images - floor is detected automatically!
                    </div>
                    """
                )

            # RIGHT COLUMN: Results
            with gr.Column(scale=5, variant="panel", elem_classes=["group-container"]):
                gr.Markdown("### 👁️ Preview")
                
                result_output = gr.Image(
                    label="Final Result",
                    type="numpy",
                    height=500,
                    interactive=False
                )
                
                with gr.Accordion("Show Mask Details", open=False):
                    mask_output = gr.Image(
                        label="Detected Floor Mask",
                        type="numpy",
                        height=250,
                        interactive=False
                    )

        # Footer / Instructions
        with gr.Row():
            gr.Markdown(
                """
                <div style="text-align: center; color: #64748b; margin-top: 2rem; font-size: 0.9rem;">
                Powered by DeepLabV3+ Semantic Segmentation • Built with Gradio
                </div>
                """
            )

        # interactions
        process_btn.click(
            fn=visualize_floor,
            inputs=[room_input, texture_input, blend_slider],
            outputs=[result_output, mask_output]
        )

    return demo


def main():
    """Main entry point."""
    print("=" * 50)
    print("Flooring Visualizer - Starting Up")
    print("=" * 50)

    try:
        initialize_model()
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Warning: {e}")
        print("Model will be loaded on first use.")

    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7861,
        share=False,
        show_error=True
    )


if __name__ == "__main__":
    main()
