#!/usr/bin/env python3
"""
Tetralemma Neural Networks (TNNs)
=================================

A novel neural architecture grounded in non-classical logic, where each neuron
outputs a 4-valued tetralemmic state reflecting the poles of Madhyamaka Catuṣkoṭi logic:
- Affirmation (P)
- Negation (¬P) 
- Both (P ∧ ¬P)
- Neither (¬(P ∨ ¬P))
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import json
from dataclasses import dataclass
from enum import Enum
import matplotlib.pyplot as plt

class TetralemmaPole(Enum):
    """The four poles of Catuṣkoṭi logic"""
    AFFIRMATION = 0    # P
    NEGATION = 1       # ¬P
    BOTH = 2           # P ∧ ¬P
    NEITHER = 3        # ¬(P ∨ ¬P)

@dataclass
class Tetrapoint:
    """A tetrapoint representing the four logical positions"""
    a: float          # Affirmation
    not_a: float      # Negation
    both: float       # Both true and false
    neither: float    # Neither true nor false
    
    def to_tensor(self) -> torch.Tensor:
        return torch.tensor([self.a, self.not_a, self.both, self.neither])
    
    def __str__(self):
        return f"({self.a:.3f}, {self.not_a:.3f}, {self.both:.3f}, {self.neither:.3f})"

class TetralemmaNeuron(nn.Module):
    """
    A Tetralemma Neuron that outputs a 4-dimensional tetrapoint
    instead of a scalar value.
    """
    
    def __init__(self, activation_fn: str = "sigmoid"):
        super().__init__()
        self.activation_fn = activation_fn
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass producing a tetrapoint output.
        
        Args:
            x: Input tensor of shape [batch_size, features]
            
        Returns:
            tetrapoint: Tensor of shape [batch_size, 4] representing (a, ¬a, a∧¬a, ¬(a∨¬a))
        """
        if self.activation_fn == "sigmoid":
            a = torch.sigmoid(x)
        elif self.activation_fn == "tanh":
            a = torch.tanh(x)
        elif self.activation_fn == "relu":
            a = torch.relu(x)
        else:
            a = x
            
        # Ensure a is in [0, 1] for proper tetralemma interpretation
        a = torch.clamp(a, 0, 1)
        
        # Calculate the four poles
        not_a = 1 - a
        both = a * not_a  # Contradiction peak when a ≈ 0.5
        neither = torch.clamp(1 - (a + not_a), 0, 1)  # Null ground
        
        # Stack into tetrapoint
        tetrapoint = torch.stack([a, not_a, both, neither], dim=-1)
        return tetrapoint

class TetralemmaLayer(nn.Module):
    """
    A dense layer that outputs tetrapoints for each neuron.
    """
    
    def __init__(self, in_features: int, out_features: int, activation_fn: str = "sigmoid"):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.tetralemma_neurons = nn.ModuleList([
            TetralemmaNeuron(activation_fn) for _ in range(out_features)
        ])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Tetralemma layer.
        
        Args:
            x: Input tensor of shape [batch_size, in_features]
            
        Returns:
            output: Tensor of shape [batch_size, out_features, 4]
        """
        # Linear transformation
        z = self.linear(x)
        
        # Apply tetralemma neurons to each output
        tetrapoints = []
        for i, neuron in enumerate(self.tetralemma_neurons):
            tetrapoint = neuron(z[:, i:i+1])  # Shape: [batch_size, 4]
            tetrapoints.append(tetrapoint)
            
        # Stack all tetrapoints and remove singleton dimension
        output = torch.stack(tetrapoints, dim=1).squeeze(-2)  # Shape: [batch_size, out_features, 4]
        return output

class ContradictionProduct(nn.Module):
    """
    Implements the contradiction product (⊗) for fusing tetrapoints.
    """
    
    def __init__(self):
        super().__init__()
        
    def forward(self, t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
        """
        Apply contradiction product between two tetrapoints.
        
        Args:
            t1: First tetrapoint tensor of shape [batch_size, 4]
            t2: Second tetrapoint tensor of shape [batch_size, 4]
            
        Returns:
            result: Fused tetrapoint tensor of shape [batch_size, 4]
        """
        # Define polar product function
        def polar_product(p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
            # Map values to tetralemma logic
            # 1 (EXPRESSED), 0 (SUPPRESSED), -1 (INAPPLICABLE), -2 (EMPTY)
            
            # Convert to tetralemma values
            def to_tetralemma_value(x: torch.Tensor) -> torch.Tensor:
                # Map [0, 1] to tetralemma values
                # High values (≈1) -> EXPRESSED (1)
                # Low values (≈0) -> SUPPRESSED (0)
                # Mid values (≈0.5) -> INAPPLICABLE (-1)
                # Very low values -> EMPTY (-2)
                
                # Simple mapping for demonstration
                expressed = (x > 0.7).float()
                suppressed = (x < 0.3).float()
                inapplicable = ((x >= 0.3) & (x <= 0.7)).float()
                empty = (x < 0.1).float()
                
                return expressed * 1 + suppressed * 0 + inapplicable * (-1) + empty * (-2)
            
            p1_tet = to_tetralemma_value(p1)
            p2_tet = to_tetralemma_value(p2)
            
            # Apply contradiction product rules
            result = torch.zeros_like(p1)
            
            # EMPTY absorption
            empty_mask = (p1_tet == -2) | (p2_tet == -2)
            result[empty_mask] = -2
            
            # INAPPLICABLE rules
            inapplicable_mask = (p1_tet == -1) | (p2_tet == -1)
            result[inapplicable_mask] = -1
            
            # SUPPRESSION rules
            suppressed_mask = (p1_tet == 0) | (p2_tet == 0)
            result[suppressed_mask] = 0
            
            # EXPRESSED rules
            expressed_mask = (p1_tet == 1) & (p2_tet == 1)
            result[expressed_mask] = 1
            
            # Convert back to [0, 1] range
            result = torch.clamp((result + 2) / 3, 0, 1)
            
            return result
        
        # Apply polar product to each component
        result_a = polar_product(t1[:, 0], t2[:, 0])
        result_not_a = polar_product(t1[:, 1], t2[:, 1])
        result_both = polar_product(t1[:, 2], t2[:, 2])
        result_neither = polar_product(t1[:, 3], t2[:, 3])
        
        return torch.stack([result_a, result_not_a, result_both, result_neither], dim=-1)

class TetralemmaAttention(nn.Module):
    """
    Attention mechanism using Tetralemma logic for contradiction-aware attention.
    """
    
    def __init__(self, hidden_dim: int, num_heads: int = 8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.contradiction_product = ContradictionProduct()
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Tetralemma attention forward pass.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, hidden_dim]
            mask: Optional attention mask
            
        Returns:
            output: Attended output with same shape as input
        """
        batch_size, seq_len, _ = x.shape
        
        # Project to query, key, value
        Q = self.query_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        K = self.key_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        V = self.value_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Transpose for attention computation
        Q = Q.transpose(1, 2)  # [batch_size, num_heads, seq_len, head_dim]
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        # Compute attention scores using tetralemma logic
        # Convert to tetrapoints for attention computation
        Q_tetrapoints = self._to_tetrapoints(Q)
        K_tetrapoints = self._to_tetrapoints(K)
        
        # Compute attention using contradiction product
        attention_scores = self._tetralemma_attention_scores(Q_tetrapoints, K_tetrapoints)
        
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))
        
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # Apply attention to values
        output = torch.matmul(attention_weights, V)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        output = self.output_proj(output)
        
        return output
    
    def _to_tetrapoints(self, x: torch.Tensor) -> torch.Tensor:
        """Convert tensor to tetrapoints for tetralemma computation."""
        # Simple conversion: use sigmoid for affirmation, derive others
        a = torch.sigmoid(x)
        not_a = 1 - a
        both = a * not_a
        neither = torch.clamp(1 - (a + not_a), 0, 1)
        
        return torch.stack([a, not_a, both, neither], dim=-1)
    
    def _tetralemma_attention_scores(self, Q: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
        """Compute attention scores using tetralemma logic."""
        # For simplicity, use a weighted combination of tetrapoint components
        # In practice, you might want more sophisticated tetralemma-specific scoring
        
        # Extract affirmation components for primary attention
        Q_a = Q[:, :, :, 0]  # [batch_size, num_heads, seq_len, head_dim]
        K_a = K[:, :, :, 0]
        
        # Standard attention with affirmation
        scores_a = torch.matmul(Q_a, K_a.transpose(-2, -1)) / np.sqrt(self.head_dim)
        
        # Add contradiction awareness
        Q_both = Q[:, :, :, 2]  # Contradiction component
        K_both = K[:, :, :, 2]
        scores_contradiction = torch.matmul(Q_both, K_both.transpose(-2, -1))
        
        # Combine scores: primary attention + contradiction awareness
        scores = scores_a + 0.1 * scores_contradiction
        
        return scores

class TetralemmaLoss(nn.Module):
    """
    Loss functions designed for Tetralemma Neural Networks.
    """
    
    def __init__(self, loss_type: str = "categorical"):
        super().__init__()
        self.loss_type = loss_type
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute Tetralemma loss.
        
        Args:
            predictions: Predicted tetrapoints of shape [batch_size, num_classes, 4]
            targets: Target tetrapoints or class labels
            
        Returns:
            loss: Computed loss value
        """
        if self.loss_type == "categorical":
            return self._categorical_loss(predictions, targets)
        elif self.loss_type == "contradiction":
            return self._contradiction_loss(predictions, targets)
        elif self.loss_type == "emptiness":
            return self._emptiness_loss(predictions, targets)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
    
    def _categorical_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Loss for categorical classification with tetrapoint outputs."""
        # Convert targets to tetrapoints if they're class indices
        if targets.dim() == 1:
            # Targets are class indices, convert to tetrapoints
            batch_size, num_classes, _ = predictions.shape
            target_tetrapoints = torch.zeros(batch_size, num_classes, 4, device=predictions.device)
            
            for i, target_idx in enumerate(targets):
                # Create a tetrapoint with high affirmation for the target class
                target_tetrapoints[i, target_idx, 0] = 1.0  # Affirmation
                target_tetrapoints[i, target_idx, 1] = 0.0  # Negation
                target_tetrapoints[i, target_idx, 2] = 0.0  # Both
                target_tetrapoints[i, target_idx, 3] = 0.0  # Neither
        else:
            target_tetrapoints = targets
        
        # MSE loss between predicted and target tetrapoints
        loss = F.mse_loss(predictions, target_tetrapoints)
        return loss
    
    def _contradiction_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Loss that encourages or suppresses contradictions."""
        # Extract contradiction components (both)
        contradiction_scores = predictions[:, :, 2]  # [batch_size, num_classes]
        
        # Encourage contradictions for certain classes (e.g., philosophical paradoxes)
        # This is a simplified version - you might want more sophisticated logic
        contradiction_target = torch.zeros_like(contradiction_scores)
        
        # For demonstration: encourage contradictions for class 2 (paradox class)
        if contradiction_scores.shape[1] > 2:
            contradiction_target[:, 2] = 0.5  # Target contradiction level
        
        loss = F.mse_loss(contradiction_scores, contradiction_target)
        return loss
    
    def _emptiness_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Loss that drives evolution toward emptiness (Ψ)."""
        # Encourage all components to approach 0 (emptiness)
        emptiness_target = torch.zeros_like(predictions)
        
        # Weighted loss: encourage emptiness but allow some structure
        emptiness_weight = 0.1
        loss = emptiness_weight * F.mse_loss(predictions, emptiness_target)
        
        return loss

class TetralemmaNeuralNetwork(nn.Module):
    """
    A complete Tetralemma Neural Network architecture.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, 
                 num_layers: int = 2, use_attention: bool = True):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_layers = num_layers
        
        # Input projection to tetralemma space
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Tetralemma layers
        self.tetralemma_layers = nn.ModuleList([
            TetralemmaLayer(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])
        
        # Tetrapoint to hidden projection for attention
        self.tetrapoint_to_hidden = nn.Linear(4, hidden_dim)
        # Hidden to tetrapoint projection after attention
        self.hidden_to_tetrapoint = nn.Linear(hidden_dim, 4)
        
        # Tetralemma attention (optional)
        self.use_attention = use_attention
        if use_attention:
            self.attention = TetralemmaAttention(hidden_dim)
        
        # Output layer
        self.output_layer = TetralemmaLayer(hidden_dim, num_classes)
        
        # Contradiction product for fusion
        self.contradiction_product = ContradictionProduct()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Tetralemma Neural Network.
        
        Args:
            x: Input tensor of shape [batch_size, input_dim]
            
        Returns:
            output: Output tetrapoints of shape [batch_size, num_classes, 4]
        """
        # Project input
        x = self.input_proj(x)
        
        # Pass through tetralemma layers
        for layer in self.tetralemma_layers:
            # Apply tetralemma layer
            tetrapoints = layer(x)  # [batch_size, hidden_dim, 4]
            
            # Apply attention if enabled
            if self.use_attention:
                # Project tetrapoints to hidden_dim for attention
                batch_size, seq_len, _ = tetrapoints.shape
                tetrapoints_proj = self.tetrapoint_to_hidden(tetrapoints)  # [batch, seq_len, hidden_dim]
                attended = self.attention(tetrapoints_proj)
                # Project back to tetrapoint space for next layer
                tetrapoints = self.hidden_to_tetrapoint(attended)  # [batch, seq_len, 4]
            
            # Use affirmation component for next layer input
            x = tetrapoints[:, :, 0]  # Take affirmation component
        
        # Final output layer
        output = self.output_layer(x)  # [batch_size, num_classes, 4]
        
        return output
    
    def predict_pole(self, x: torch.Tensor, pole: TetralemmaPole) -> torch.Tensor:
        """
        Predict the specific tetralemma pole.
        
        Args:
            x: Input tensor
            pole: Which pole to predict
            
        Returns:
            predictions: Predictions for the specified pole
        """
        output = self.forward(x)
        return output[:, :, pole.value]
    
    def get_contradiction_score(self, x: torch.Tensor) -> torch.Tensor:
        """Get the contradiction score for each class."""
        output = self.forward(x)
        return output[:, :, 2]  # Both component

# Dataset for philosophical text classification
class PhilosophicalTextDataset(Dataset):
    """Dataset for philosophical text classification using Tetralemma logic."""
    
    def __init__(self, texts: List[str], labels: List[int], max_length: int = 100):
        self.texts = texts
        self.labels = labels
        self.max_length = max_length
        
        # Simple vocabulary for demonstration
        self.vocab = self._build_vocab()
        
    def _build_vocab(self) -> Dict[str, int]:
        """Build vocabulary from texts."""
        vocab = {"<PAD>": 0, "<UNK>": 1}
        word_freq = {}
        
        for text in self.texts:
            words = text.lower().split()
            for word in words:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Add most frequent words to vocabulary
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        for word, _ in sorted_words[:1000]:  # Limit vocabulary size
            if word not in vocab:
                vocab[word] = len(vocab)
        
        return vocab
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Tokenize and encode text
        words = text.lower().split()[:self.max_length]
        encoded = [self.vocab.get(word, self.vocab["<UNK>"]) for word in words]
        
        # Pad to max_length
        if len(encoded) < self.max_length:
            encoded += [self.vocab["<PAD>"]] * (self.max_length - len(encoded))
        
        return torch.tensor(encoded, dtype=torch.long), torch.tensor(label, dtype=torch.long)

def train_tetralemma_network():
    """Train a Tetralemma Neural Network on philosophical text classification."""
    
    print("🧠 Training Tetralemma Neural Network")
    print("=" * 50)
    
    # Sample philosophical texts for demonstration
    texts = [
        "The self exists as a permanent entity",
        "The self does not exist at all", 
        "The self both exists and does not exist",
        "The question of self-existence is meaningless",
        "Reality is fundamentally real",
        "Reality is an illusion",
        "Reality is both real and illusory",
        "The concept of reality is empty",
        "Truth is absolute and unchanging",
        "Truth is relative and contextual",
        "Truth is both absolute and relative",
        "Truth transcends all categories",
        "Free will is real and meaningful",
        "Free will is an illusion",
        "Free will both exists and does not exist",
        "The concept of free will is empty",
    ]
    
    # Labels: 0=affirmation, 1=negation, 2=both, 3=neither
    labels = [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]
    
    # Create dataset
    dataset = PhilosophicalTextDataset(texts, labels)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # Initialize model
    model = TetralemmaNeuralNetwork(
        input_dim=1000,  # Vocabulary size
        hidden_dim=64,
        num_classes=4,   # Four tetralemma poles
        num_layers=2,
        use_attention=True
    )
    
    # Loss and optimizer
    criterion = TetralemmaLoss(loss_type="categorical")
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # For visualization
    losses = []
    avg_affirmation = []
    avg_negation = []
    avg_both = []
    avg_neither = []
    num_epochs = 50
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        epoch_affirmation = []
        epoch_negation = []
        epoch_both = []
        epoch_neither = []
        
        for batch_idx, (data, targets) in enumerate(dataloader):
            # Convert to one-hot encoding for input
            batch_size = data.shape[0]
            input_onehot = torch.zeros(batch_size, 1000)
            for i in range(batch_size):
                for j in range(data.shape[1]):
                    if data[i, j] < 1000:
                        input_onehot[i, data[i, j]] = 1
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(input_onehot)
            
            # Compute loss
            loss = criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            # Track tetrapoint averages
            epoch_affirmation.append(outputs[:, :, 0].detach().mean().item())
            epoch_negation.append(outputs[:, :, 1].detach().mean().item())
            epoch_both.append(outputs[:, :, 2].detach().mean().item())
            epoch_neither.append(outputs[:, :, 3].detach().mean().item())
        
        losses.append(total_loss/len(dataloader))
        avg_affirmation.append(np.mean(epoch_affirmation))
        avg_negation.append(np.mean(epoch_negation))
        avg_both.append(np.mean(epoch_both))
        avg_neither.append(np.mean(epoch_neither))
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(dataloader):.4f}")
    
    print("\n✅ Training completed!")
    
    # Plot learning dynamics
    plt.figure(figsize=(10, 6))
    plt.plot(losses, label='Loss', color='black', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Tetralemma Neural Network Training Loss')
    plt.legend()
    plt.show()
    
    plt.figure(figsize=(10, 6))
    plt.plot(avg_affirmation, label='Affirmation (P)')
    plt.plot(avg_negation, label='Negation (¬P)')
    plt.plot(avg_both, label='Both (P∧¬P)')
    plt.plot(avg_neither, label='Neither (¬(P∨¬P))')
    plt.xlabel('Epoch')
    plt.ylabel('Average Value')
    plt.title('Tetrapoint Component Dynamics During Training')
    plt.legend()
    plt.show()
    
    # Test the model
    model.eval()
    with torch.no_grad():
        test_text = "The mind is both real and unreal"
        test_dataset = PhilosophicalTextDataset([test_text], [2])
        test_data, _ = test_dataset[0]
        
        # Convert to one-hot
        test_onehot = torch.zeros(1, 1000)
        for j in range(test_data.shape[0]):
            if test_data[j] < 1000:
                test_onehot[0, test_data[j]] = 1
        
        prediction = model(test_onehot)
        
        print(f"\n🔍 Test Prediction for: '{test_text}'")
        print("Tetrapoint output:")
        for i, pole in enumerate(TetralemmaPole):
            value = prediction[0, i, :].detach().numpy()
            print(f"  {pole.name}: {value}")
        
        # Get contradiction score
        contradiction_score = model.get_contradiction_score(test_onehot)
        print(f"Contradiction Score: {contradiction_score[0].detach().numpy()}")
    
    return model

def demonstrate_tetralemma_operations():
    """Demonstrate various Tetralemma neural network operations."""
    
    print("\n🧪 Tetralemma Neural Network Operations")
    print("=" * 50)
    
    # Create a simple tetralemma layer
    layer = TetralemmaLayer(in_features=10, out_features=5)
    
    # Test input
    x = torch.randn(3, 10)  # Batch size 3, 10 features
    print(f"Input shape: {x.shape}")
    
    # Forward pass
    output = layer(x)
    print(f"Output shape: {output.shape}")
    print(f"Output tetrapoints:")
    for i in range(3):
        for j in range(5):
            tetrapoint = output[i, j].detach().numpy()
            print(f"  Sample {i}, Neuron {j}: {tetrapoint}")
    
    # Test contradiction product
    contradiction_product = ContradictionProduct()
    t1 = torch.tensor([[0.8, 0.2, 0.1, 0.1], [0.3, 0.7, 0.2, 0.1]])
    t2 = torch.tensor([[0.9, 0.1, 0.05, 0.05], [0.4, 0.6, 0.2, 0.1]])
    
    fused = contradiction_product(t1, t2)
    print(f"\nContradiction Product:")
    print(f"t1: {t1.detach().numpy()}")
    print(f"t2: {t2.detach().numpy()}")
    print(f"t1 ⊗ t2: {fused.detach().numpy()}")
    
    # Test attention mechanism
    attention = TetralemmaAttention(hidden_dim=64)
    x_attention = torch.randn(2, 10, 64)  # Batch, seq_len, hidden_dim
    attended = attention(x_attention)
    print(f"\nAttention output shape: {attended.shape}")

if __name__ == "__main__":
    # Demonstrate basic operations
    demonstrate_tetralemma_operations()
    
    # Train the network
    model = train_tetralemma_network()
    
    print("\n🎉 Tetralemma Neural Network demonstration completed!")
    print("This implementation shows how non-classical logic can be integrated into deep learning.") 