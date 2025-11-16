from torch import nn
import torch
from types import SimpleNamespace

from .nGPT_model import Block
from .pooler import Pooler

class LinearLayer(nn.Module):
    def __init__(self, config):
        super(LinearLayer, self).__init__()
        input_dim = config.input_dim
        output_dim = config.output_dim
        dropout = config.dropout
        pooler_type = config.pooler_type

        self.pooler = Pooler(pooler_type)
        setattr(self, pooler_type, self.pooler)

        self.linear = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_dim, output_dim),
        )

    def forward(self, embedding, attention_mask):
        pooled = self.pooler(attention_mask, embedding)
        output = self.linear(pooled)  # project to probing head
        return output

    def encode(self, embedding, attention_mask):
        """
        Returns the output of the linear layer.
        """
        return self.forward(embedding, attention_mask)

class MLP2Layer(nn.Module):
    def __init__(self, config):
        super(MLP2Layer, self).__init__()
        input_dim = config.input_dim
        intermediate_dim = config.intermediate_dim
        bottleneck_dim = config.bottleneck_dim
        output_dim = config.output_dim
        dropout = config.dropout
        pooler_type = config.pooler_type

        self.pooler = Pooler(pooler_type)
        setattr(self, pooler_type, self.pooler)

        self.compressor = nn.Sequential(
            nn.Linear(input_dim, intermediate_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_dim, bottleneck_dim),
            nn.ReLU()
        )

        self.probing_head = nn.Linear(bottleneck_dim, output_dim)

    def forward(self, embedding, attention_mask):
        pooled = self.pooler(attention_mask, embedding)
        compressed = self.compressor(pooled)   # compress embedding
        output = self.probing_head(compressed)    # project to probing head
        return output

    def encode(self, embedding, attention_mask):
        """
        Returns the output of the MLP2 layer.
        """
        return self.forward(embedding, attention_mask)
    
class ContrastiveClassifier(nn.Module):
    def __init__(self, config):
        super(ContrastiveClassifier, self).__init__()
        input_dim = config.input_dim
        intermediate_dim = config.intermediate_dim
        output_dim = config.output_dim
        dropout = config.dropout
        pooler_type = config.pooler_type

        self.pooler = Pooler(pooler_type)
        setattr(self, pooler_type, self.pooler)

        self.embedder = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_dim, intermediate_dim),
        )
        self.classifier = nn.Sequential(
            nn.Linear(intermediate_dim, output_dim),
        )

    def forward(self, x, attention_mask):
        pooled = self.pooler(attention_mask, x)
        embedding = self.embedder(pooled)
        logits = self.classifier(embedding)
        # 出力次元が1なら二値分類用のシグモイド、
        # それ以外は多クラス用のソフトマックスで確率を返す
        if logits.size(-1) == 1:
            probs = torch.sigmoid(logits)
        else:
            probs = torch.softmax(logits, dim=-1)
        return embedding, probs

    def encode(self, x, attention_mask):
        """
        Returns the embedding from the embedder.
        """
        pooled = self.pooler(attention_mask, x)
        return self.embedder(pooled)


class _GPTBlockClassifierBase(nn.Module):
    """
    Shared logic for GPT-style transformer block classifiers.

    Takes a pooled embedding, runs it through a single transformer block borrowed
    from the GPT/nGPT implementation, and projects it with a learnable head.
    """

    def __init__(self, config, use_ngpt: bool):
        super().__init__()
        self.config = config
        self.use_ngpt = use_ngpt

        block_config = SimpleNamespace(
            n_embd=config.input_dim,
            n_head=config.num_heads,
            base_scale=config.base_scale,
            use_nGPT=1 if use_ngpt else 0,
            bias=config.bias,
        )
        # iblock is unused inside Block, so we can fix it at 0.
        self.transformer = Block(block_config, iblock=0)

        pooler_type = config.pooler_type
        self.pooler = Pooler(pooler_type)

    def _ensure_sequence_dim(self, embedding: torch.Tensor) -> torch.Tensor:
        if embedding.dim() == 2:
            return embedding.unsqueeze(1)
        if embedding.dim() == 3:
            return embedding
        raise ValueError(
            f"Expected 2D or 3D tensor for GPT-style classifier, got shape {embedding.shape}"
        )

    def forward(self, embedding: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        sequence = self._ensure_sequence_dim(embedding)
        transformed = self.transformer(sequence)
        return self.pooler(attention_mask, transformed)

    def encode(self, embedding: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.forward(embedding, attention_mask)


class GPTClassifier(_GPTBlockClassifierBase):
    """Classifier that uses a standard GPT transformer block."""

    def __init__(self, config):
        super().__init__(config, use_ngpt=False)


class nGPTClassifier(_GPTBlockClassifierBase):
    """Classifier that uses the custom nGPT transformer block."""

    def __init__(self, config):
        super().__init__(config, use_ngpt=True)
    
