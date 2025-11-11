from torch import nn
import torch

class LinearLayer(nn.Module):
    def __init__(self, config):
        super(LinearLayer, self).__init__()
        input_dim = config.input_dim
        output_dim = config.output_dim
        dropout = config.dropout
        
        self.linear = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_dim, output_dim),
        )
        
    def forward(self, embedding):
        output = self.linear(embedding)  # project to probing head
        return output
    
    def encode(self, embedding):
        """
        Returns the output of the linear layer.
        """
        return self.forward(embedding)

class MLP2Layer(nn.Module):
    def __init__(self, config):
        super(MLP2Layer, self).__init__()
        input_dim = config.input_dim
        intermediate_dim = config.intermediate_dim
        bottleneck_dim = config.bottleneck_dim
        output_dim = config.output_dim
        dropout = config.dropout
        
        self.compressor = nn.Sequential(
            nn.Linear(input_dim, intermediate_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_dim, bottleneck_dim),
            nn.ReLU()
        )
        
        self.probing_head = nn.Linear(bottleneck_dim, output_dim)
        
    def forward(self, embedding):
        compressed = self.compressor(embedding)   # compress embedding
        output = self.probing_head(compressed)    # project to probing head
        return output
    
    def encode(self, embedding):
        """
        Returns the output of the MLP2 layer.
        """
        return self.forward(embedding)
    
class ContrastiveClassifier(nn.Module):
    def __init__(self, config):
        super(ContrastiveClassifier, self).__init__()
        input_dim = config.input_dim
        intermediate_dim = config.intermediate_dim
        output_dim = config.output_dim
        dropout = config.dropout
        
        self.embedder = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_dim, intermediate_dim),
        )
        self.classifier = nn.Sequential(
            nn.Linear(intermediate_dim, output_dim),
        )

    def forward(self, x):
        embedding = self.embedder(x)
        logits = self.classifier(embedding)
        # 出力次元が1なら二値分類用のシグモイド、
        # それ以外は多クラス用のソフトマックスで確率を返す
        if logits.size(-1) == 1:
            probs = torch.sigmoid(logits)
        else:
            probs = torch.softmax(logits, dim=-1)
        return embedding, probs
    
    def encode(self, x):
        """
        Returns the embedding from the embedder.
        """
        return self.embedder(x)
    
