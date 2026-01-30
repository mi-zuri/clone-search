"""Streamlit application for face analysis: search, inpainting, and Grad-CAM visualization."""

import os
import sys

# Fix OpenMP conflict between PyTorch and FAISS on macOS
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Add project root to path for imports when running via streamlit
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch
from PIL import Image

from src.data.augmentations import get_val_augmentations
from src.models.encoder import FaceEncoder
from src.search.engine import FaceSearchEngine

# ============================================================================
# Page Configuration
# ============================================================================

st.set_page_config(
    page_title="Face Analysis App",
    page_icon="👤",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================================
# Cached Resource Loaders
# ============================================================================


@st.cache_resource
def get_device() -> torch.device:
    """Get best available device: MPS > CUDA > CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@st.cache_resource
def load_encoder(checkpoint_path: str = "checkpoints/best_encoder.pt") -> FaceEncoder:
    """Load the trained FaceEncoder model."""
    device = get_device()

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model = FaceEncoder(
        embedding_dim=64,
        projection_dim=64,
        num_attributes=40,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    return model


@st.cache_resource
def load_search_engine(
    gallery_path: str = "checkpoints/gallery_index.npz",
) -> FaceSearchEngine:
    """Load the FAISS-based face search engine."""
    return FaceSearchEngine(gallery_path)


@st.cache_resource
def get_transform():
    """Get validation transforms for preprocessing images."""
    return get_val_augmentations(256)


# ============================================================================
# Helper Functions
# ============================================================================


def load_and_preprocess_image(
    path_or_file, transform
) -> tuple[torch.Tensor, Image.Image]:
    """Load and preprocess an image for the model.

    Args:
        path_or_file: Either a file path (str/Path) or an UploadedFile
        transform: Torchvision transform pipeline

    Returns:
        Tuple of (preprocessed tensor, original PIL image)
    """
    if isinstance(path_or_file, (str, Path)):
        image = Image.open(path_or_file).convert("RGB")
    else:
        # Streamlit UploadedFile
        image = Image.open(path_or_file).convert("RGB")

    # Keep original for display
    original = image.copy()

    # Apply transforms
    tensor = transform(image)

    return tensor, original


def apply_heatmap_overlay(
    image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.5
) -> np.ndarray:
    """Blend a heatmap onto an image using jet colormap.

    Args:
        image: RGB image array (H, W, 3), uint8
        heatmap: Heatmap array (H, W) normalized to [0, 1]
        alpha: Blend factor (0 = image only, 1 = heatmap only)

    Returns:
        Blended image as uint8 array
    """
    # Apply jet colormap
    cmap = plt.get_cmap("jet")
    heatmap_colored = cmap(heatmap)[:, :, :3]  # Remove alpha channel
    heatmap_colored = (heatmap_colored * 255).astype(np.uint8)

    # Resize heatmap to match image if needed
    if heatmap_colored.shape[:2] != image.shape[:2]:
        from PIL import Image as PILImage
        from PIL.Image import Resampling

        heatmap_pil = PILImage.fromarray(heatmap_colored)
        heatmap_pil = heatmap_pil.resize(
            (image.shape[1], image.shape[0]), Resampling.BILINEAR
        )
        heatmap_colored = np.array(heatmap_pil)

    # Blend
    blended = (1 - alpha) * image.astype(np.float32) + alpha * heatmap_colored.astype(
        np.float32
    )
    blended = np.clip(blended, 0, 255).astype(np.uint8)

    return blended


# ============================================================================
# Grad-CAM Implementation
# ============================================================================


class GradCAM:
    """Grad-CAM visualization for FaceEncoder.

    Captures activations and gradients from the last MobileNetV2 block
    to generate class activation maps showing which regions influenced
    the model's output.
    """

    def __init__(self, model: FaceEncoder, device: torch.device):
        self.model = model
        self.device = device
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None

        # Hook into last feature block (MobileNetV2 block 18)
        self.target_layer = model.features[-1]
        self._register_hooks()

    def _register_hooks(self):
        """Register forward and backward hooks on the target layer."""

        def forward_hook(_module, _input, output):
            self.activations = output.detach()

        def backward_hook(_module, _grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(
        self,
        image_tensor: torch.Tensor,
        target: str = "embedding",
        attribute_idx: Optional[int] = None,
    ) -> tuple[np.ndarray, dict]:
        """Generate Grad-CAM heatmap for the given target.

        Args:
            image_tensor: Preprocessed image tensor (1, 3, 256, 256)
            target: Either "embedding" for overall face or "attribute" for specific attribute
            attribute_idx: Attribute index if target is "attribute"

        Returns:
            Tuple of (heatmap array normalized [0,1], info dict with predictions)
        """
        self.model.eval()
        image_tensor = image_tensor.to(self.device)

        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)

        # Enable gradients for this forward pass
        image_tensor.requires_grad_(True)

        # Forward pass
        embedding, attr_logits = self.model(image_tensor)

        # Compute target for backward pass
        if target == "embedding":
            # Use L2 norm of embedding as target
            target_output = embedding.norm(p=2, dim=1)
        else:
            # Use specific attribute logit
            target_output = attr_logits[:, attribute_idx]

        # Backward pass
        self.model.zero_grad()
        target_output.backward()

        # Compute Grad-CAM
        # Global average pooling of gradients
        assert self.gradients is not None, "Gradients not captured"
        assert self.activations is not None, "Activations not captured"
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

        # Weighted combination of activation maps
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, H, W)

        # ReLU and normalize
        cam = torch.relu(cam)
        cam = cam.squeeze().cpu().numpy()

        # Normalize to [0, 1]
        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min())

        # Resize to 256x256
        cam_pil = Image.fromarray((cam * 255).astype(np.uint8))
        cam_pil = cam_pil.resize((256, 256), Image.Resampling.BILINEAR)
        cam = np.array(cam_pil).astype(np.float32) / 255.0

        # Collect prediction info
        attr_probs = torch.sigmoid(attr_logits).squeeze().cpu().detach().numpy()
        info = {
            "embedding_norm": float(embedding.norm(p=2, dim=1).item()),
            "attribute_probs": {
                name: float(prob)
                for name, prob in zip(FaceSearchEngine.ATTRIBUTE_NAMES, attr_probs)
            },
        }

        return cam, info


# ============================================================================
# Grad-CAM Display Component
# ============================================================================


def render_gradcam_analysis(image_path: str, image_label: str):
    """Render Grad-CAM analysis for a selected image.

    Args:
        image_path: Path to the image file
        image_label: Label to display (e.g., "Query" or "Result #1")
    """
    encoder = load_encoder()
    transform = get_transform()
    device = get_device()

    st.markdown(f"#### Grad-CAM Analysis: {image_label}")

    # Load image
    tensor, original_image = load_and_preprocess_image(image_path, transform)

    # Target selection
    target_options = ["Embedding (overall face)"] + list(FaceSearchEngine.ATTRIBUTE_NAMES)
    target = st.selectbox(
        "Visualization Target",
        target_options,
        key=f"gradcam_target_{image_label}",
    )

    if st.button("Generate Grad-CAM", key=f"gradcam_btn_{image_label}"):
        with st.spinner("Computing Grad-CAM..."):
            # Initialize GradCAM (need fresh instance to avoid hook accumulation)
            gradcam = GradCAM(encoder, device)

            # Determine target
            if target == "Embedding (overall face)":
                target_type = "embedding"
                attr_idx = None
            else:
                target_type = "attribute"
                attr_idx = FaceSearchEngine.ATTRIBUTE_NAMES.index(target)

            # Generate heatmap
            heatmap, info = gradcam.generate(tensor, target=target_type, attribute_idx=attr_idx)

            # Create visualization
            original_array = np.array(original_image.resize((256, 256)))
            overlay = apply_heatmap_overlay(original_array, heatmap, alpha=0.5)

            # Display results in 3 columns
            col1, col2, col3 = st.columns(3)

            with col1:
                st.write("**Original**")
                st.image(original_image, width="stretch")

            with col2:
                st.write("**Heatmap**")
                fig, ax = plt.subplots(figsize=(4, 4))
                im = ax.imshow(heatmap, cmap="jet")
                ax.axis("off")
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                st.pyplot(fig)
                plt.close()

            with col3:
                st.write("**Overlay**")
                st.image(overlay, width="stretch")

            # Show prediction for selected attribute
            if target_type == "attribute":
                prob = info["attribute_probs"][target]
                st.metric(f"Predicted: {target}", f"{prob:.1%}")

            # Show top attributes
            with st.expander("Top Predicted Attributes"):
                sorted_attrs = sorted(
                    info["attribute_probs"].items(),
                    key=lambda x: x[1],
                    reverse=True,
                )[:10]
                for attr, prob in sorted_attrs:
                    st.write(f"- **{attr}**: {prob:.1%}")


# ============================================================================
# Tab 1: Face Search with Integrated Grad-CAM
# ============================================================================


def render_face_search_tab():
    """Render the Face Search tab UI with integrated Grad-CAM."""
    st.header("Face Search")
    st.write("Search for similar faces in the gallery, then analyze any result with Grad-CAM.")

    # Load resources
    try:
        engine = load_search_engine()
        encoder = load_encoder()
        transform = get_transform()
        device = get_device()
    except Exception as e:
        st.error(f"Failed to load resources: {e}")
        st.info("Make sure checkpoints/best_encoder.pt and checkpoints/gallery_index.npz exist.")
        return

    # Query mode selection
    query_mode = st.radio(
        "Query Mode",
        ["Random from Gallery", "Filter by Attributes", "Upload Image"],
        horizontal=True,
    )

    query_embedding = None
    query_image = None
    query_path = None

    if query_mode == "Random from Gallery":
        if st.button("Pick Random Face"):
            idx = np.random.randint(0, len(engine))
            query_path = engine.paths[idx]
            query_embedding = engine.get_embedding_by_index(idx)

            # Load image for display
            _, query_image = load_and_preprocess_image(query_path, transform)

            st.session_state["query_embedding"] = query_embedding
            st.session_state["query_image"] = query_image
            st.session_state["query_path"] = query_path
            st.session_state["search_results"] = None  # Clear old results

        # Retrieve from session state
        if "query_embedding" in st.session_state:
            query_embedding = st.session_state["query_embedding"]
            query_image = st.session_state["query_image"]
            query_path = st.session_state["query_path"]

    elif query_mode == "Filter by Attributes":
        col1, col2 = st.columns(2)

        with col1:
            positive_attrs = st.multiselect(
                "Must Have (positive attributes)",
                FaceSearchEngine.ATTRIBUTE_NAMES,
                key="positive_attrs",
            )

        with col2:
            negative_attrs = st.multiselect(
                "Must NOT Have (negative attributes)",
                [a for a in FaceSearchEngine.ATTRIBUTE_NAMES if a not in positive_attrs],
                key="negative_attrs",
            )

        if st.button("Find Matching Face"):
            # Filter gallery by attributes
            matches = []
            for idx in range(len(engine)):
                attr_probs = engine.attributes[idx]
                attr_dict = {
                    name: prob
                    for name, prob in zip(FaceSearchEngine.ATTRIBUTE_NAMES, attr_probs)
                }

                # Check positive attributes (prob > 0.5)
                pos_ok = all(attr_dict[a] > 0.5 for a in positive_attrs)
                # Check negative attributes (prob < 0.5)
                neg_ok = all(attr_dict[a] < 0.5 for a in negative_attrs)

                if pos_ok and neg_ok:
                    matches.append(idx)

            if matches:
                idx = np.random.choice(matches)
                query_path = engine.paths[idx]
                query_embedding = engine.get_embedding_by_index(idx)
                _, query_image = load_and_preprocess_image(query_path, transform)

                st.session_state["query_embedding"] = query_embedding
                st.session_state["query_image"] = query_image
                st.session_state["query_path"] = query_path
                st.session_state["search_results"] = None
                st.success(f"Found {len(matches)} matching faces, picked one at random.")
            else:
                st.warning("No faces match the selected attributes. Try different filters.")

        # Retrieve from session state
        if "query_embedding" in st.session_state:
            query_embedding = st.session_state["query_embedding"]
            query_image = st.session_state["query_image"]
            query_path = st.session_state.get("query_path")

    else:  # Upload Image
        uploaded_file = st.file_uploader(
            "Upload a face image", type=["jpg", "jpeg", "png"]
        )

        if uploaded_file is not None:
            tensor, query_image = load_and_preprocess_image(uploaded_file, transform)

            # Get embedding from encoder
            with torch.no_grad():
                tensor = tensor.unsqueeze(0).to(device)
                query_embedding, _ = encoder(tensor)
                query_embedding = query_embedding.cpu().numpy().squeeze()

            st.session_state["query_embedding"] = query_embedding
            st.session_state["query_image"] = query_image
            st.session_state["query_path"] = None  # No file path for uploaded
            st.session_state["uploaded_file"] = uploaded_file
            st.session_state["search_results"] = None

        if "query_embedding" in st.session_state:
            query_embedding = st.session_state["query_embedding"]
            query_image = st.session_state["query_image"]
            query_path = st.session_state.get("query_path")

    # Search options
    st.subheader("Search Options")
    col1, col2, col3 = st.columns(3)

    with col1:
        k = st.slider("Number of results (k)", min_value=5, max_value=50, value=10)

    with col2:
        rerank_positive = st.multiselect(
            "Re-rank: MUST have",
            FaceSearchEngine.ATTRIBUTE_NAMES,
            key="rerank_positive",
            help="Boost results with these attributes",
        )

    with col3:
        # Exclude already selected positive attrs from negative options
        available_negative = [a for a in FaceSearchEngine.ATTRIBUTE_NAMES if a not in rerank_positive]
        rerank_negative = st.multiselect(
            "Re-rank: must NOT have",
            available_negative,
            key="rerank_negative",
            help="Boost results WITHOUT these attributes",
        )

    # Build attribute filters for re-ranking
    attribute_filters = None
    if rerank_positive or rerank_negative:
        attribute_filters = {}
        for attr in rerank_positive:
            attribute_filters[attr] = True
        for attr in rerank_negative:
            attribute_filters[attr] = False

    # Display query and results
    if query_embedding is not None and query_image is not None:
        st.subheader("Query Image")
        st.image(query_image, width=200)
        if query_path:
            st.caption(f"Source: {query_path}")

        # Perform search
        results = engine.search(query_embedding, k=k, attribute_filters=attribute_filters)
        st.session_state["search_results"] = results

        st.subheader(f"Top {len(results)} Similar Faces")
        st.caption("Click a result number below to analyze it with Grad-CAM")

        # Display results in a grid (5 columns)
        cols = st.columns(5)
        for i, result in enumerate(results):
            col = cols[i % 5]
            with col:
                try:
                    img = Image.open(result["path"]).convert("RGB")
                    st.image(img, width="stretch")

                    # Score display with result number
                    if attribute_filters:
                        st.caption(
                            f"**#{i+1}** | Sim: {result['similarity']:.3f}"
                        )
                    else:
                        st.caption(f"**#{i+1}** | Sim: {result['similarity']:.3f}")

                except Exception:
                    st.error(f"Could not load: {result['path']}")

        # ================================================================
        # Grad-CAM Analysis Section
        # ================================================================
        st.divider()
        st.subheader("Grad-CAM Visualization")
        st.write("Select an image to visualize what the model focuses on.")

        # Build selection options
        selection_options = []
        selection_paths = {}

        if query_path:
            selection_options.append("Query Image")
            selection_paths["Query Image"] = query_path

        for i, result in enumerate(results):
            label = f"Result #{i+1} (sim: {result['similarity']:.3f})"
            selection_options.append(label)
            selection_paths[label] = result["path"]

        if selection_options:
            selected = st.selectbox(
                "Select image to analyze",
                selection_options,
                key="gradcam_selection",
            )

            if selected and selected in selection_paths:
                render_gradcam_analysis(selection_paths[selected], selected)


# ============================================================================
# Tab 2: Inpainting (Stub)
# ============================================================================


def render_inpainting_tab():
    """Render the Inpainting tab UI (stub implementation)."""
    st.header("Face Inpainting")

    st.info(
        "Coming Soon: Face inpainting with semantic mask selection. "
        "This feature will allow you to remove and reconstruct facial regions "
        "like eyes, nose, or mouth using a trained UNet model."
    )

    # Disabled UI skeleton
    st.subheader("Upload Image")
    st.file_uploader(
        "Choose a face image",
        type=["jpg", "jpeg", "png"],
        disabled=True,
        key="inpaint_upload",
    )

    st.subheader("Select Region to Inpaint")
    st.selectbox(
        "Mask Region",
        ["Eyes", "Nose", "Mouth", "Hair", "Full Face"],
        disabled=True,
        key="inpaint_region",
    )

    st.button("Generate Inpainting", disabled=True, key="inpaint_button")

    # Placeholder columns
    st.subheader("Results")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.write("**Original**")
        st.empty()

    with col2:
        st.write("**Masked**")
        st.empty()

    with col3:
        st.write("**Reconstructed**")
        st.empty()


# ============================================================================
# Main Application
# ============================================================================


def main():
    """Main application entry point."""
    st.title("Face Analysis Application")

    # Sidebar
    with st.sidebar:
        st.header("About")
        st.write(
            "This application provides face analysis tools:\n\n"
            "1. **Face Search**: Find similar faces in an 80k gallery and "
            "visualize model attention with Grad-CAM\n"
            "2. **Inpainting**: Reconstruct masked facial regions (coming soon)"
        )

        st.header("System Info")
        device = get_device()
        st.write(f"**Device:** {device}")
        st.write(f"**PyTorch:** {torch.__version__}")

    # Create tabs (now just 2)
    tab1, tab2 = st.tabs(["Face Search", "Inpainting"])

    with tab1:
        render_face_search_tab()

    with tab2:
        render_inpainting_tab()


if __name__ == "__main__":
    main()
