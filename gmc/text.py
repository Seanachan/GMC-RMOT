"""Frozen sentence encoder for the referring expressions."""

import torch

DEFAULT_TEXT_ENCODER = "all-MiniLM-L6-v2"


class TextEncoder:
    """Thin wrapper around a sentence-transformers model (MiniLM by default,
    384-D). The model is never fine-tuned."""

    def __init__(self, model_name: str = DEFAULT_TEXT_ENCODER, device: str = "cpu") -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name).to(device)
        self.device = device
        self.model_name = model_name
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

    def encode(self, texts, batch_size: int = 32, show_progress_bar: bool = False) -> torch.Tensor:
        """str -> (1, D) tensor; list[str] -> (N, D) tensor."""
        with torch.no_grad():
            emb = self.model.encode(
                texts,
                batch_size=batch_size,
                convert_to_tensor=True,
                show_progress_bar=show_progress_bar,
            )
        if isinstance(texts, str) and emb.ndim == 1:
            emb = emb.unsqueeze(0)
        return emb.to(self.device)
