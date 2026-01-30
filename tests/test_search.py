"""Tests for face search engine."""

import numpy as np
import pytest

from src.data.dataset import CelebADataset
from src.search.engine import FaceSearchEngine


@pytest.fixture
def mock_gallery(tmp_path):
    """Create a mock gallery NPZ file for testing."""
    n_samples = 100
    embedding_dim = 64
    num_attributes = 40

    # Generate random L2-normalized embeddings
    embeddings = np.random.randn(n_samples, embedding_dim).astype(np.float32)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    # Generate random attribute probabilities in [0, 1]
    attributes = np.random.rand(n_samples, num_attributes).astype(np.float32)

    # Generate paths and sources
    paths = np.array([f"data/test/{i:05d}.jpg" for i in range(n_samples)])
    sources = np.array(["celeba"] * 60 + ["ffhq"] * 40)

    # Save to NPZ
    gallery_path = tmp_path / "test_gallery.npz"
    np.savez(
        gallery_path,
        embeddings=embeddings,
        attributes=attributes,
        paths=paths,
        sources=sources,
    )

    return gallery_path, {
        "embeddings": embeddings,
        "attributes": attributes,
        "paths": paths,
        "sources": sources,
    }


class TestNPZFormat:
    """Test NPZ file format and data shapes."""

    def test_embeddings_shape(self, mock_gallery):
        """Embeddings should be (N, 64) float32."""
        gallery_path, data = mock_gallery
        loaded = np.load(gallery_path)

        assert loaded["embeddings"].shape == (100, 64)
        assert loaded["embeddings"].dtype == np.float32

    def test_attributes_shape(self, mock_gallery):
        """Attributes should be (N, 40) float32."""
        gallery_path, data = mock_gallery
        loaded = np.load(gallery_path)

        assert loaded["attributes"].shape == (100, 40)
        assert loaded["attributes"].dtype == np.float32

    def test_paths_and_sources(self, mock_gallery):
        """Paths and sources should be string arrays of length N."""
        gallery_path, data = mock_gallery
        loaded = np.load(gallery_path)

        assert len(loaded["paths"]) == 100
        assert len(loaded["sources"]) == 100

    def test_embeddings_are_normalized(self, mock_gallery):
        """Embeddings should be L2 normalized (unit norm)."""
        gallery_path, data = mock_gallery
        loaded = np.load(gallery_path)

        norms = np.linalg.norm(loaded["embeddings"], axis=1)
        assert np.allclose(norms, 1.0, atol=1e-5)

    def test_attributes_in_valid_range(self, mock_gallery):
        """Attribute probabilities should be in [0, 1]."""
        gallery_path, data = mock_gallery
        loaded = np.load(gallery_path)

        assert loaded["attributes"].min() >= 0.0
        assert loaded["attributes"].max() <= 1.0


class TestFaceSearchEngine:
    """Test FaceSearchEngine class."""

    def test_engine_initialization(self, mock_gallery, capsys):
        """Engine should load gallery and report stats."""
        gallery_path, _ = mock_gallery
        engine = FaceSearchEngine(str(gallery_path))

        assert len(engine) == 100
        captured = capsys.readouterr()
        assert "100 faces" in captured.out

    def test_attribute_names(self, mock_gallery):
        """Engine should expose all 40 CelebA attribute names."""
        gallery_path, _ = mock_gallery
        engine = FaceSearchEngine(str(gallery_path))

        assert len(engine.ATTRIBUTE_NAMES) == 40
        assert engine.ATTRIBUTE_NAMES == CelebADataset.ATTRIBUTE_NAMES

    def test_get_embedding_by_index(self, mock_gallery):
        """Should retrieve correct embedding by index."""
        gallery_path, data = mock_gallery
        engine = FaceSearchEngine(str(gallery_path))

        embedding = engine.get_embedding_by_index(5)
        assert embedding.shape == (64,)
        assert np.allclose(embedding, data["embeddings"][5])


class TestSearchResults:
    """Test search functionality."""

    def test_search_returns_k_results(self, mock_gallery):
        """Search should return exactly k results."""
        gallery_path, _ = mock_gallery
        engine = FaceSearchEngine(str(gallery_path))

        query = np.random.randn(64).astype(np.float32)
        query /= np.linalg.norm(query)

        results = engine.search(query, k=5)
        assert len(results) == 5

        results = engine.search(query, k=10)
        assert len(results) == 10

    def test_search_result_keys(self, mock_gallery):
        """Each result should have required keys."""
        gallery_path, _ = mock_gallery
        engine = FaceSearchEngine(str(gallery_path))

        query = np.random.randn(64).astype(np.float32)
        query /= np.linalg.norm(query)

        results = engine.search(query, k=5)

        required_keys = {"path", "similarity", "final_score", "attributes", "source"}
        for result in results:
            assert set(result.keys()) == required_keys

    def test_search_result_types(self, mock_gallery):
        """Result values should have correct types."""
        gallery_path, _ = mock_gallery
        engine = FaceSearchEngine(str(gallery_path))

        query = np.random.randn(64).astype(np.float32)
        query /= np.linalg.norm(query)

        results = engine.search(query, k=1)
        result = results[0]

        assert isinstance(result["path"], str)
        assert isinstance(result["similarity"], float)
        assert isinstance(result["final_score"], float)
        assert isinstance(result["attributes"], dict)
        assert isinstance(result["source"], str)
        assert result["source"] in ("celeba", "ffhq")

    def test_similarity_descending_order(self, mock_gallery):
        """Results should be sorted by similarity (descending)."""
        gallery_path, _ = mock_gallery
        engine = FaceSearchEngine(str(gallery_path))

        query = np.random.randn(64).astype(np.float32)
        query /= np.linalg.norm(query)

        results = engine.search(query, k=20)
        similarities = [r["similarity"] for r in results]

        # Should be in descending order
        assert similarities == sorted(similarities, reverse=True)

    def test_similarity_in_valid_range(self, mock_gallery):
        """Cosine similarity should be in [-1, 1]."""
        gallery_path, _ = mock_gallery
        engine = FaceSearchEngine(str(gallery_path))

        query = np.random.randn(64).astype(np.float32)
        query /= np.linalg.norm(query)

        results = engine.search(query, k=10)

        for result in results:
            assert -1.0 <= result["similarity"] <= 1.0

    def test_query_1d_and_2d(self, mock_gallery):
        """Search should accept both 1D and 2D query arrays."""
        gallery_path, _ = mock_gallery
        engine = FaceSearchEngine(str(gallery_path))

        query = np.random.randn(64).astype(np.float32)
        query /= np.linalg.norm(query)

        # 1D query
        results_1d = engine.search(query, k=5)

        # 2D query
        results_2d = engine.search(query.reshape(1, -1), k=5)

        # Should produce same results
        assert len(results_1d) == len(results_2d)
        for r1, r2 in zip(results_1d, results_2d):
            assert r1["path"] == r2["path"]
            assert np.isclose(r1["similarity"], r2["similarity"])


class TestAttributeFiltering:
    """Test attribute-based filtering and re-ranking."""

    def test_filter_re_ranks_results(self, mock_gallery):
        """Attribute filtering should re-rank results by final_score."""
        gallery_path, _ = mock_gallery
        engine = FaceSearchEngine(str(gallery_path))

        query = np.random.randn(64).astype(np.float32)
        query /= np.linalg.norm(query)

        # Search with filter
        results = engine.search(query, k=10, attribute_filters={"Smiling": True})

        # final_scores should be in descending order
        final_scores = [r["final_score"] for r in results]
        assert final_scores == sorted(final_scores, reverse=True)

    def test_filter_affects_final_score(self, mock_gallery):
        """Filtering should modify final_score based on attribute confidence."""
        gallery_path, data = mock_gallery
        engine = FaceSearchEngine(str(gallery_path))

        query = np.random.randn(64).astype(np.float32)
        query /= np.linalg.norm(query)

        # Without filter
        results_no_filter = engine.search(query, k=10)

        # With filter
        results_filtered = engine.search(query, k=10, attribute_filters={"Smiling": True})

        # final_score should differ from similarity when filtered
        for result in results_filtered:
            # final_score = similarity * P(Smiling)
            # Unless P(Smiling) = 1, final_score < similarity
            smiling_prob = result["attributes"]["Smiling"]
            expected_final = result["similarity"] * smiling_prob
            assert np.isclose(result["final_score"], expected_final, atol=1e-6)

    def test_filter_false_uses_complement(self, mock_gallery):
        """Filter with False should use (1 - prob) as confidence."""
        gallery_path, _ = mock_gallery
        engine = FaceSearchEngine(str(gallery_path))

        query = np.random.randn(64).astype(np.float32)
        query /= np.linalg.norm(query)

        results = engine.search(query, k=10, attribute_filters={"Eyeglasses": False})

        for result in results:
            glasses_prob = result["attributes"]["Eyeglasses"]
            expected_final = result["similarity"] * (1.0 - glasses_prob)
            assert np.isclose(result["final_score"], expected_final, atol=1e-6)

    def test_multiple_filters_multiply(self, mock_gallery):
        """Multiple filters should multiply confidence scores."""
        gallery_path, _ = mock_gallery
        engine = FaceSearchEngine(str(gallery_path))

        query = np.random.randn(64).astype(np.float32)
        query /= np.linalg.norm(query)

        results = engine.search(
            query, k=10, attribute_filters={"Smiling": True, "Male": False}
        )

        for result in results:
            smiling_prob = result["attributes"]["Smiling"]
            male_prob = result["attributes"]["Male"]
            expected = result["similarity"] * smiling_prob * (1.0 - male_prob)
            assert np.isclose(result["final_score"], expected, atol=1e-6)

    def test_unknown_attribute_raises(self, mock_gallery):
        """Unknown attribute name should raise ValueError."""
        gallery_path, _ = mock_gallery
        engine = FaceSearchEngine(str(gallery_path))

        query = np.random.randn(64).astype(np.float32)
        query /= np.linalg.norm(query)

        with pytest.raises(ValueError, match="Unknown attribute"):
            engine.search(query, k=5, attribute_filters={"NotAnAttribute": True})

    def test_no_filter_equal_scores(self, mock_gallery):
        """Without filter, final_score should equal similarity."""
        gallery_path, _ = mock_gallery
        engine = FaceSearchEngine(str(gallery_path))

        query = np.random.randn(64).astype(np.float32)
        query /= np.linalg.norm(query)

        results = engine.search(query, k=10)

        for result in results:
            assert result["final_score"] == result["similarity"]


class TestAttributeDict:
    """Test attribute dictionary in results."""

    def test_attributes_has_all_names(self, mock_gallery):
        """Result attributes should contain all 40 attribute names."""
        gallery_path, _ = mock_gallery
        engine = FaceSearchEngine(str(gallery_path))

        query = np.random.randn(64).astype(np.float32)
        query /= np.linalg.norm(query)

        results = engine.search(query, k=1)
        attrs = results[0]["attributes"]

        assert len(attrs) == 40
        for name in CelebADataset.ATTRIBUTE_NAMES:
            assert name in attrs

    def test_attributes_are_floats(self, mock_gallery):
        """All attribute values should be Python floats."""
        gallery_path, _ = mock_gallery
        engine = FaceSearchEngine(str(gallery_path))

        query = np.random.randn(64).astype(np.float32)
        query /= np.linalg.norm(query)

        results = engine.search(query, k=1)
        attrs = results[0]["attributes"]

        for name, value in attrs.items():
            assert isinstance(value, float), f"{name} is {type(value)}"

    def test_attributes_in_valid_range(self, mock_gallery):
        """Attribute probabilities should be in [0, 1]."""
        gallery_path, _ = mock_gallery
        engine = FaceSearchEngine(str(gallery_path))

        query = np.random.randn(64).astype(np.float32)
        query /= np.linalg.norm(query)

        results = engine.search(query, k=10)

        for result in results:
            for name, prob in result["attributes"].items():
                assert 0.0 <= prob <= 1.0, f"{name}={prob} out of range"
