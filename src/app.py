"""Streamlit application for face search and inpainting."""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import torch
import numpy as np
from PIL import Image
import cv2

from src.models.encoder import FaceEncoder
from src.models.unet import UNet
from src.search.engine import SearchEngine
from src.data.augmentations import get_val_transforms


# Page config
st.set_page_config(
    page_title="Face Search & Inpainting",
    page_icon="🔍",
    layout="wide"
)


@st.cache_resource
def load_models():
    """Load encoder and U-Net models."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

    # Load encoder
    encoder = FaceEncoder(embedding_dim=128, num_attributes=40)
    encoder_path = Path("checkpoints/encoder_best.pth")
    if encoder_path.exists():
        checkpoint = torch.load(encoder_path, map_location=device)
        encoder.load_state_dict(checkpoint['model_state_dict'])
        st.sidebar.success("Encoder loaded!")
    else:
        st.sidebar.warning("Encoder checkpoint not found. Using untrained model.")
    encoder = encoder.to(device)
    encoder.eval()

    # Load U-Net
    unet = UNet(in_channels=4, out_channels=3)
    unet_path = Path("checkpoints/unet_best.pth")
    if unet_path.exists():
        checkpoint = torch.load(unet_path, map_location=device)
        unet.load_state_dict(checkpoint['model_state_dict'])
        st.sidebar.success("U-Net loaded!")
    else:
        st.sidebar.warning("U-Net checkpoint not found. Using untrained model.")
    unet = unet.to(device)
    unet.eval()

    return encoder, unet, device


@st.cache_resource
def load_search_engine(_encoder, _device):
    """Load search engine with gallery index."""
    engine = SearchEngine(_encoder, _device, use_faiss=True)

    index_path = Path("checkpoints/gallery_index.npy")
    if index_path.exists():
        engine.load_index(str(index_path))
        st.sidebar.success(f"Gallery loaded: {len(engine.embeddings)} images")
    else:
        st.sidebar.warning("Gallery index not found. Search will not work.")

    return engine


def preprocess_image(image: Image.Image, size: int = 256) -> torch.Tensor:
    """Preprocess image for model input."""
    transform = get_val_transforms(size)
    img_array = np.array(image.convert('RGB'))
    transformed = transform(image=img_array)
    return transformed['image'].unsqueeze(0)


def generate_face_region_mask(image: np.ndarray, region: str) -> np.ndarray:
    """Generate approximate mask for face region.

    Uses simple heuristics based on typical face proportions.
    """
    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    # Face center approximation
    cx, cy = w // 2, h // 2

    if region == "Eyes":
        # Eyes region: upper-middle area
        eye_h = h // 6
        eye_w = w // 2
        y1 = int(cy - eye_h * 1.2)
        y2 = int(cy - eye_h * 0.2)
        x1 = int(cx - eye_w // 2)
        x2 = int(cx + eye_w // 2)
        mask[y1:y2, x1:x2] = 255

    elif region == "Nose":
        # Nose region: center area
        nose_h = h // 4
        nose_w = w // 5
        y1 = int(cy - nose_h // 2)
        y2 = int(cy + nose_h // 2)
        x1 = int(cx - nose_w // 2)
        x2 = int(cx + nose_w // 2)
        mask[y1:y2, x1:x2] = 255

    elif region == "Mouth":
        # Mouth region: lower-middle area
        mouth_h = h // 6
        mouth_w = w // 3
        y1 = int(cy + mouth_h * 0.5)
        y2 = int(cy + mouth_h * 1.8)
        x1 = int(cx - mouth_w // 2)
        x2 = int(cx + mouth_w // 2)
        mask[y1:y2, x1:x2] = 255

    elif region == "Forehead":
        # Forehead region: upper area
        forehead_h = h // 5
        forehead_w = w // 2
        y1 = int(cy - h * 0.35)
        y2 = int(cy - h * 0.15)
        x1 = int(cx - forehead_w // 2)
        x2 = int(cx + forehead_w // 2)
        mask[y1:y2, x1:x2] = 255

    else:  # Random rectangle
        rect_h = np.random.randint(40, min(80, h // 3))
        rect_w = np.random.randint(40, min(80, w // 3))
        y1 = np.random.randint(h // 4, h * 3 // 4 - rect_h)
        x1 = np.random.randint(w // 4, w * 3 // 4 - rect_w)
        mask[y1:y1+rect_h, x1:x1+rect_w] = 255

    return mask


def apply_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Apply mask to image (set masked pixels to white)."""
    masked = image.copy()
    masked[mask > 0] = 255
    return masked


def inpaint_image(unet, image: Image.Image, mask: np.ndarray, device) -> np.ndarray:
    """Run inpainting on masked image."""
    # Resize image and mask to 256x256
    img_resized = image.resize((256, 256), Image.BILINEAR)
    img_array = np.array(img_resized).astype(np.float32) / 255.0

    mask_resized = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)
    mask_array = mask_resized.astype(np.float32) / 255.0

    # Create masked image
    masked_img = img_array * (1 - mask_array[:, :, None])

    # Convert to tensor
    img_tensor = torch.from_numpy(masked_img).permute(2, 0, 1).unsqueeze(0).to(device)
    mask_tensor = torch.from_numpy(mask_array).unsqueeze(0).unsqueeze(0).to(device)

    # Concatenate and run model
    inputs = torch.cat([img_tensor, mask_tensor], dim=1)

    with torch.no_grad():
        output = unet(inputs)

    # Convert back to numpy
    result = output[0].permute(1, 2, 0).cpu().numpy()
    result = (result * 255).astype(np.uint8)

    # Resize back to original size
    original_size = image.size
    result_resized = cv2.resize(result, original_size, interpolation=cv2.INTER_LANCZOS4)

    return result_resized


def main():
    st.title("Face Search & Inpainting")

    # Load models
    encoder, unet, device = load_models()
    search_engine = load_search_engine(encoder, device)

    # Sidebar info
    st.sidebar.header("Model Info")
    st.sidebar.write(f"Device: {device}")

    # Tabs
    tab1, tab2, tab3 = st.tabs(["Face Search", "Inpainting", "Grad-CAM (Optional)"])

    # Tab 1: Face Search
    with tab1:
        st.header("Find Similar Faces")
        st.write("Upload an image to find similar faces in the gallery.")

        uploaded_file = st.file_uploader("Upload a face image", type=['jpg', 'jpeg', 'png'], key='search')

        if uploaded_file is not None:
            image = Image.open(uploaded_file)

            col1, col2 = st.columns([1, 2])

            with col1:
                st.subheader("Query Image")
                st.image(image, use_container_width=True)

            with col2:
                st.subheader("Search Results")

                if search_engine.embeddings is not None:
                    # Preprocess and search
                    img_tensor = preprocess_image(image)
                    indices, similarities, paths = search_engine.search(img_tensor.to(device), k=5)

                    # Display results
                    result_cols = st.columns(5)
                    for i, (idx, sim, path) in enumerate(zip(indices, similarities, paths)):
                        with result_cols[i]:
                            if os.path.exists(path):
                                result_img = Image.open(path)
                                st.image(result_img, caption=f"Sim: {sim:.3f}", use_container_width=True)
                            else:
                                st.write(f"#{idx}: {sim:.3f}")
                else:
                    st.warning("Gallery index not loaded. Please build the gallery index first.")
                    st.code("python -m src.search.engine --model checkpoints/encoder_best.pth --gallery data/FFHQ")

    # Tab 2: Inpainting
    with tab2:
        st.header("Face Inpainting")
        st.write("Upload an image and select a region to remove and inpaint.")

        uploaded_file = st.file_uploader("Upload a face image", type=['jpg', 'jpeg', 'png'], key='inpaint')

        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert('RGB')
            img_array = np.array(image)

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Original Image")
                st.image(image, use_container_width=True)

                region = st.selectbox(
                    "Select region to remove",
                    ["Eyes", "Nose", "Mouth", "Forehead", "Random Rectangle"]
                )

                if st.button("Generate Mask"):
                    mask = generate_face_region_mask(img_array, region)
                    st.session_state['mask'] = mask
                    st.session_state['original_image'] = image

            with col2:
                if 'mask' in st.session_state:
                    mask = st.session_state['mask']
                    masked_img = apply_mask(img_array, mask)

                    st.subheader("Masked Image")
                    st.image(masked_img, use_container_width=True)

                    if st.button("Inpaint"):
                        with st.spinner("Inpainting..."):
                            result = inpaint_image(unet, image, mask, device)
                            st.session_state['inpaint_result'] = result

            if 'inpaint_result' in st.session_state:
                st.subheader("Inpainting Result")
                result_cols = st.columns(3)
                with result_cols[0]:
                    st.image(image, caption="Original", use_container_width=True)
                with result_cols[1]:
                    if 'mask' in st.session_state:
                        masked_img = apply_mask(img_array, st.session_state['mask'])
                        st.image(masked_img, caption="Masked", use_container_width=True)
                with result_cols[2]:
                    st.image(st.session_state['inpaint_result'], caption="Inpainted", use_container_width=True)

    # Tab 3: Grad-CAM (Optional)
    with tab3:
        st.header("Grad-CAM Visualization")
        st.write("Visualize what the encoder focuses on when analyzing faces.")

        uploaded_file = st.file_uploader("Upload a face image", type=['jpg', 'jpeg', 'png'], key='gradcam')

        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert('RGB')

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Original Image")
                st.image(image, use_container_width=True)

            with col2:
                st.subheader("Grad-CAM Heatmap")

                if st.button("Generate Grad-CAM"):
                    try:
                        from pytorch_grad_cam import GradCAM
                        from pytorch_grad_cam.utils.image import show_cam_on_image

                        # Get the last conv layer
                        target_layer = encoder.features[-1]

                        # Custom target for the encoder (use embedding magnitude)
                        class EmbeddingTarget:
                            def __call__(self, model_output):
                                return model_output['embedding'].norm(dim=1)

                        cam = GradCAM(model=encoder, target_layers=[target_layer])

                        # Preprocess image
                        img_tensor = preprocess_image(image).to(device)
                        img_np = np.array(image.resize((256, 256))) / 255.0

                        # Generate CAM
                        grayscale_cam = cam(input_tensor=img_tensor, targets=[EmbeddingTarget()])
                        grayscale_cam = grayscale_cam[0, :]

                        # Overlay on image
                        visualization = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)

                        st.image(visualization, use_container_width=True)

                    except ImportError:
                        st.error("grad-cam not installed. Run: pip install grad-cam")
                    except Exception as e:
                        st.error(f"Error generating Grad-CAM: {str(e)}")


if __name__ == "__main__":
    main()
