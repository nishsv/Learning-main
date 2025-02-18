
# Transformer Architecture: From Tokenization to Decoder-Only Models

## 1. **Tokenization**
Tokenization is the process of breaking down raw text into smaller units such as words, sub-words, or characters. Libraries like **BPE**, **SentencePiece**, and **WordPiece** are commonly used for this purpose. Tokenization helps in converting raw text into machine-understandable format by assigning an integer ID to each token.

Example:
- Sentence: "I went to school today"
- Tokens: ["I", "went", "to", "school", "today"]
- After tokenization: ["I", "went", "to", "school", "today"] → [1, 2, 3, 4, 5]

## 2. **Embedding Layer**
The list of token IDs is passed into the **embedding layer**, which is essentially a lookup table that converts each token ID into its corresponding vector representation. Each token has an associated vector (e.g., 300 dimensions), and the embedding matrix shape will be **[vocab size, embedding size]**.

- The embedding layer is a mini neural network trained to capture **semantic meaning** of words and the relationship between them.

- Example: 
  - Token "went" → Embedding Vector: [0.34, -0.56, ...]

## 3. **Positional Encoding**
Transformers process words in parallel rather than sequentially, so to inject **sequential information**, we add **positional encoding** to the embedding. The positional encoding tells the model **where** each token is located in the sequence.

- Positional encoding matrix has the same shape as the embeddings: **[tokens, embedding size]**.
- The final input to the transformer is a **sum of embeddings and positional encodings**.

## 4. **Transformer Encoder**

### Structure:
- **Input Shape**: [tokens, embedding size]
- **Self-Attention Layer**: Learns to focus on relevant parts of the input sequence. It uses **Query (Q)**, **Key (K)**, and **Value (V)** matrices to calculate attention.
  
  **Why three matrices?**  
  - **Query** and **Key** are used to calculate attention scores (how much focus one token should give to another).  
  - **Value** represents the actual content that the attention mechanism uses.

- **Add & Norm**: After each self-attention and feed-forward layer, **layer normalization** is applied to stabilize training.

- **Feed-Forward Network (FFN)**: This layer applies non-linear transformations, typically two fully connected layers with an activation function.

- **Final Shape**: After encoder processing, the output has the shape **[tokens, embedding size]**.

## 5. **Transformer Decoder**

### Structure:
The decoder is similar to the encoder but with an additional layer of **cross-attention**. In addition to self-attention, it takes the output of the encoder to help generate the final output.

- **Self-Attention Layer**: Uses the decoder’s own previously predicted tokens.
- **Cross-Attention Layer**: Takes **Key** and **Value** from the encoder to align the output to the input context.

- **Add & Norm** and **Feed-Forward Network (FFN)**: Same as in the encoder.

### Decoder-Only Models:
In decoder-only models, like **GPT**, the model generates text **autoregressively** based solely on previously generated tokens, without any external encoder context. The model uses **self-attention** to decide the next word based on prior tokens.

## 6. **Contrastive Loss in Document Ranking**

In document ranking tasks, the model learns to differentiate between relevant and irrelevant documents given a query.
 
- **Input**: Query and documents are passed through the same model.
- **Output**: The target labels represent relevance scores, e.g., `[1, 0, 0]` for relevant and irrelevant documents.
- **Loss Function**: Contrastive loss, which compares the model's predicted scores for each document against the true relevance.

## Official Transformer Diagram

Here is the official diagram that illustrates the transformer architecture:
[Transformer Model Diagram](https://jalammar.github.io/images/transformer_architecture.jpg)

---

### Summary of Data Flow:

1. **Tokenization** → Convert text into tokens.
2. **Embedding Layer** → Convert tokens into embeddings.
3. **Positional Encoding** → Add positional information.
4. **Encoder** → Self-attention, Add & Norm, Feed-forward networks.
5. **Decoder** → Self-attention, Cross-attention, Add & Norm, Feed-forward networks.
6. **Output Generation** → Decoder generates text based on learned context and relationships.
