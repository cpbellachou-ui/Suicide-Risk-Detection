# Mathematical Documentation - Suicide Risk Detection

## Overview
This document provides detailed mathematical formulations for the suicide risk detection system, covering both traditional machine learning (TF-IDF + Logistic Regression) and deep learning (BERT) approaches.

## 1. Logistic Regression Mathematics

### 1.1 Sigmoid Function
The sigmoid function maps any real number to a value between 0 and 1:

```
σ(z) = 1 / (1 + e^(-z))
```

Where:
- `z` is the linear combination of features: `z = θ₀ + θ₁x₁ + θ₂x₂ + ... + θₙxₙ`
- `θ` are the model parameters (weights)
- `x` are the input features

### 1.2 Hypothesis Function
The hypothesis function for binary classification:

```
h_θ(x) = σ(θᵀx) = 1 / (1 + e^(-θᵀx))
```

This gives the probability that the input belongs to class 1 (suicide risk).

### 1.3 Binary Cross-Entropy Loss
The loss function for binary classification:

```
L(θ) = -[y log(h_θ(x)) + (1-y) log(1 - h_θ(x))]
```

For a dataset with m samples:

```
J(θ) = -(1/m) Σᵢ₌₁ᵐ [y⁽ⁱ⁾ log(h_θ(x⁽ⁱ⁾)) + (1-y⁽ⁱ⁾) log(1 - h_θ(x⁽ⁱ⁾))]
```

### 1.4 Gradient Descent Update Rule
The parameters are updated using gradient descent:

```
θⱼ := θⱼ - α ∂J(θ)/∂θⱼ
```

Where:
- `α` is the learning rate
- The partial derivative is:

```
∂J(θ)/∂θⱼ = (1/m) Σᵢ₌₁ᵐ (h_θ(x⁽ⁱ⁾) - y⁽ⁱ⁾) xⱼ⁽ⁱ⁾
```

### 1.5 Decision Boundary
The decision boundary is where the probability equals 0.5:

```
h_θ(x) = 0.5
1 / (1 + e^(-θᵀx)) = 0.5
e^(-θᵀx) = 1
θᵀx = 0
```

## 2. TF-IDF Mathematics

### 2.1 Term Frequency (TF)
The frequency of a term in a document:

```
TF(t,d) = f(t,d) / Σᵢ f(tᵢ,d)
```

Where:
- `f(t,d)` is the frequency of term t in document d
- The denominator is the total number of terms in document d

### 2.2 Inverse Document Frequency (IDF)
The inverse document frequency:

```
IDF(t) = log(N / |{d ∈ D : t ∈ d}|)
```

Where:
- `N` is the total number of documents
- `|{d ∈ D : t ∈ d}|` is the number of documents containing term t

### 2.3 TF-IDF Score
The final TF-IDF score:

```
TF-IDF(t,d) = TF(t,d) × IDF(t)
```

### 2.4 Feature Vector
For a document d, the feature vector is:

```
x = [TF-IDF(t₁,d), TF-IDF(t₂,d), ..., TF-IDF(tₙ,d)]
```

## 3. BERT Architecture Mathematics

### 3.1 Self-Attention Mechanism
The core of BERT is the self-attention mechanism:

```
Attention(Q,K,V) = softmax(QKᵀ / √dₖ) V
```

Where:
- `Q` (Query), `K` (Key), `V` (Value) are learned matrices
- `dₖ` is the dimension of the key vectors (typically 64)
- `√dₖ` is the scaling factor to prevent vanishing gradients

### 3.2 Multi-Head Attention
BERT uses multiple attention heads in parallel:

```
MultiHead(Q,K,V) = Concat(head₁, head₂, ..., headₕ) W^O
```

Where each head is:

```
headᵢ = Attention(QWᵢ^Q, KWᵢ^K, VWᵢ^V)
```

And:
- `h` is the number of attention heads (12 for BERT-base)
- `W^O` is the output projection matrix
- `Wᵢ^Q`, `Wᵢ^K`, `Wᵢ^V` are learned projection matrices for each head

### 3.3 Position Embeddings
BERT adds positional information to token embeddings:

```
E = TokenEmbedding + PositionEmbedding + SegmentEmbedding
```

### 3.4 Layer Normalization
Applied after each sublayer:

```
LayerNorm(x) = γ ⊙ (x - μ) / σ + β
```

Where:
- `μ` and `σ` are the mean and standard deviation
- `γ` and `β` are learned parameters
- `⊙` denotes element-wise multiplication

### 3.5 Feed-Forward Network
Each transformer layer contains a feed-forward network:

```
FFN(x) = max(0, xW₁ + b₁)W₂ + b₂
```

### 3.6 Classification Head
For our suicide risk detection task:

```
P(risk) = softmax(W · BERT_[CLS] + b)
```

Where:
- `BERT_[CLS]` is the [CLS] token representation (768-dimensional for BERT-base)
- `W` is a learned weight matrix (768 × 2)
- `b` is a bias vector (2-dimensional)
- `softmax` ensures probabilities sum to 1

## 4. Training Mathematics

### 4.1 Cross-Entropy Loss for BERT
For multi-class classification:

```
L = -Σᵢ₌₁ᶜ yᵢ log(pᵢ)
```

Where:
- `c` is the number of classes (2 for binary classification)
- `yᵢ` is the true label (one-hot encoded)
- `pᵢ` is the predicted probability for class i

### 4.2 Adam Optimizer
The Adam optimizer updates parameters using:

```
mₜ = β₁mₜ₋₁ + (1-β₁)gₜ
vₜ = β₂vₜ₋₁ + (1-β₂)gₜ²
m̂ₜ = mₜ / (1-β₁ᵗ)
v̂ₜ = vₜ / (1-β₂ᵗ)
θₜ = θₜ₋₁ - α · m̂ₜ / (√v̂ₜ + ε)
```

Where:
- `gₜ` is the gradient at time t
- `β₁ = 0.9`, `β₂ = 0.999` are decay rates
- `α = 2e-5` is the learning rate
- `ε = 1e-8` is a small constant for numerical stability

### 4.3 Learning Rate Scheduling
We use linear warmup and decay:

```
lr(t) = lr_min + (lr_max - lr_min) × min(t/t_warmup, 1)
```

For decay:
```
lr(t) = lr_max × (1 - t/t_total)
```

## 5. Evaluation Metrics

### 5.1 Accuracy
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

### 5.2 Precision
```
Precision = TP / (TP + FP)
```

### 5.3 Recall
```
Recall = TP / (TP + FN)
```

### 5.4 F1-Score
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

### 5.5 AUC-ROC
The Area Under the ROC Curve:

```
AUC = ∫₀¹ TPR(FPR⁻¹(u)) du
```

Where:
- `TPR` is True Positive Rate
- `FPR` is False Positive Rate
- `FPR⁻¹` is the inverse function

## 6. Attention Weight Analysis

### 6.1 Attention Weight Extraction
For a given input sequence of length L:

```
Attention_weights = softmax(QKᵀ / √dₖ) ∈ ℝ^(L×L)
```

### 6.2 Attention Aggregation
To get attention for each token:

```
Attention_i = Σⱼ Attention_weights[i,j]
```

### 6.3 Multi-Layer Attention
Average attention across all layers and heads:

```
Final_attention = (1/H) Σₕ₌₁ᴴ (1/L) Σₗ₌₁ᴸ Attention_weights^(l,h)
```

Where:
- `H` is the number of attention heads
- `L` is the number of layers
- `Attention_weights^(l,h)` is the attention matrix for layer l, head h

## 7. Mathematical Complexity

### 7.1 TF-IDF Complexity
- **Time Complexity**: O(N × M) where N is vocabulary size, M is total terms
- **Space Complexity**: O(N × D) where D is number of documents

### 7.2 Logistic Regression Complexity
- **Training Time**: O(N × D × I) where I is number of iterations
- **Prediction Time**: O(N) where N is number of features

### 7.3 BERT Complexity
- **Time Complexity**: O(L² × H × D) where L is sequence length, H is hidden size, D is model depth
- **Space Complexity**: O(L² × H) for attention matrices

## 8. Regularization Techniques

### 8.1 L2 Regularization (Logistic Regression)
```
J(θ) = J₀(θ) + λ Σᵢ θᵢ²
```

Where `λ` is the regularization parameter.

### 8.2 Dropout (BERT)
During training, randomly set some neurons to zero:

```
y = f(x ⊙ mask) / (1 - p)
```

Where:
- `mask` is a binary mask with probability `p` of being 0
- Division by `(1-p)` maintains expected value

### 8.3 Weight Decay (BERT)
```
L_total = L_task + λ Σᵢ ||Wᵢ||²
```

## 9. Mathematical Intuition

### 9.1 Why BERT Works Better
1. **Contextual Understanding**: Self-attention captures long-range dependencies
2. **Transfer Learning**: Pre-trained on massive text corpus
3. **Hierarchical Features**: Multiple layers learn features at different levels
4. **Bidirectional Context**: Unlike RNNs, processes entire sequence at once

### 9.2 Feature Importance Interpretation
- **TF-IDF**: High weights indicate terms that are frequent in one class but rare in the other
- **BERT**: Attention weights show which tokens the model focuses on for classification

### 9.3 Decision Boundary
- **Logistic Regression**: Linear decision boundary in feature space
- **BERT**: Non-linear decision boundary in high-dimensional embedding space

## 10. Implementation Notes

### 10.1 Numerical Stability
- Use log-sum-exp trick for softmax computation
- Clip gradients to prevent exploding gradients
- Use mixed precision training for efficiency

### 10.2 Memory Optimization
- Gradient checkpointing for large models
- Dynamic batching based on sequence length
- Model parallelism for very large models

This mathematical foundation provides the theoretical basis for understanding and implementing the suicide risk detection system effectively.