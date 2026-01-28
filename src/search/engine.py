"""Search engine for finding similar faces using embeddings."""

import os
import sys
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
import torch
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class SearchEngine:
    """Face search engine using cosine similarity."""

    def __init__(
        self,
        model,
        device: torch.device,
        use_faiss: bool = True
    ):
        """Initialize search engine.

        Args:
            model: Face encoder model
            device: Torch device
            use_faiss: Use FAISS for efficient search
        """
        self.model = model
        self.device = device
        self.use_faiss = use_faiss

        self.embeddings = None
        self.image_paths = None
        self.index = None

        if use_faiss:
            try:
                import faiss
                self.faiss = faiss
            except ImportError:
                print("FAISS not available, falling back to brute force search")
                self.use_faiss = False

    def build_index(self, dataset, batch_size: int = 32):
        """Build search index from dataset.

        Args:
            dataset: Dataset with images
            batch_size: Batch size for embedding extraction
        """
        self.model.eval()
        embeddings_list = []
        paths_list = []

        import platform
        num_workers = 0 if platform.system() == 'Windows' else 4

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )

        print("Extracting embeddings...")
        with torch.no_grad():
            for batch in tqdm(dataloader):
                images = batch['image'].to(self.device)
                embeddings = self.model.get_embedding(images)
                embeddings_list.append(embeddings.cpu().numpy())

                # Store paths if available
                if 'path' in batch:
                    paths_list.extend(batch['path'])
                else:
                    paths_list.extend([str(i) for i in batch['idx'].tolist()])

        self.embeddings = np.vstack(embeddings_list).astype(np.float32)
        self.image_paths = paths_list

        print(f"Built index with {len(self.embeddings)} embeddings")

        # Build FAISS index
        if self.use_faiss:
            self._build_faiss_index()

    def _build_faiss_index(self):
        """Build FAISS index for efficient search."""
        embedding_dim = self.embeddings.shape[1]

        # Use inner product (cosine similarity for normalized vectors)
        self.index = self.faiss.IndexFlatIP(embedding_dim)
        self.index.add(self.embeddings)
        print(f"Built FAISS index with {self.index.ntotal} vectors")

    def search(
        self,
        query_image: torch.Tensor,
        k: int = 5
    ) -> Tuple[List[int], List[float], List[str]]:
        """Search for similar faces.

        Args:
            query_image: Query image tensor (1, 3, H, W) or (3, H, W)
            k: Number of results to return

        Returns:
            Tuple of (indices, similarities, paths)
        """
        if query_image.dim() == 3:
            query_image = query_image.unsqueeze(0)

        self.model.eval()
        with torch.no_grad():
            query_embedding = self.model.get_embedding(query_image.to(self.device))
            query_embedding = query_embedding.cpu().numpy().astype(np.float32)

        if self.use_faiss and self.index is not None:
            # FAISS search
            similarities, indices = self.index.search(query_embedding, k)
            similarities = similarities[0].tolist()
            indices = indices[0].tolist()
        else:
            # Brute force cosine similarity
            similarities = np.dot(self.embeddings, query_embedding.T).squeeze()
            indices = np.argsort(similarities)[::-1][:k].tolist()
            similarities = similarities[indices].tolist()

        paths = [self.image_paths[i] for i in indices]

        return indices, similarities, paths

    def search_with_attributes(
        self,
        query_image: torch.Tensor,
        attribute_filters: Optional[Dict[str, bool]] = None,
        k: int = 5
    ) -> Tuple[List[int], List[float], List[str]]:
        """Search with optional attribute filtering.

        Note: This is a simplified version. Full attribute filtering would
        require storing attribute predictions for all gallery images.

        Args:
            query_image: Query image tensor
            attribute_filters: Dict of attribute_name -> required_value
            k: Number of results

        Returns:
            Tuple of (indices, similarities, paths)
        """
        # For now, just return regular search results
        # Full attribute filtering would require pre-computing attributes for gallery
        return self.search(query_image, k)

    def save_index(self, path: str):
        """Save embeddings and paths to file."""
        save_dict = {
            'embeddings': self.embeddings,
            'image_paths': self.image_paths
        }
        np.save(path, save_dict)
        print(f"Saved index to {path}")

    def load_index(self, path: str):
        """Load embeddings and paths from file."""
        save_dict = np.load(path, allow_pickle=True).item()
        self.embeddings = save_dict['embeddings'].astype(np.float32)
        self.image_paths = save_dict['image_paths']

        if self.use_faiss:
            self._build_faiss_index()

        print(f"Loaded index with {len(self.embeddings)} embeddings")


def build_gallery_index(
    model_path: str,
    gallery_path: str,
    output_path: str,
    device: Optional[torch.device] = None,
    batch_size: int = 32
):
    """Build and save gallery index.

    Args:
        model_path: Path to encoder checkpoint
        gallery_path: Path to gallery dataset (FFHQ)
        output_path: Path to save index
        device: Torch device
        batch_size: Batch size for extraction
    """
    from src.models.encoder import FaceEncoder
    from src.data.dataset import FFHQDataset

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

    # Load model
    model = FaceEncoder()
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    # Create dataset
    dataset = FFHQDataset(gallery_path)
    print(f"Gallery has {len(dataset)} images")

    # Build index
    engine = SearchEngine(model, device, use_faiss=True)
    engine.build_index(dataset, batch_size=batch_size)

    # Save
    engine.save_index(output_path)
    print(f"Gallery index saved to {output_path}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Build search index')
    parser.add_argument('--model', type=str, required=True, help='Path to encoder checkpoint')
    parser.add_argument('--gallery', type=str, required=True, help='Path to gallery dataset')
    parser.add_argument('--output', type=str, default='checkpoints/gallery_index.npy', help='Output path')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    args = parser.parse_args()

    build_gallery_index(args.model, args.gallery, args.output, batch_size=args.batch_size)
