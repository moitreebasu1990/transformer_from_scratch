# Transformer for Machine Translation

This project implements a Transformer model from scratch for machine translation from English to French using PyTorch.

## Table of Contents

1. [Introduction](#introduction)
2. [Setup](#setup)
3. [Data Preparation](#data-preparation)
4. [Model Architecture](#model-architecture)
5. [Training](#training)
6. [Evaluation](#evaluation)
7. [Usage](#usage)

## Introduction

This project demonstrates the implementation of a Transformer-based translation model. The Transformer architecture, introduced in the paper "Attention Is All You Need," has become a cornerstone in Natural Language Processing tasks.

## Setup

To run this project, you need the following dependencies:

- PyTorch
- torchtext
- NLTK
- tqdm

You can install these dependencies using pip:

```
pip install torch torchtext nltk tqdm
```

## Data Preparation

The project uses the 'opus_books' dataset for English to French translation. The data is processed to create vocabularies for both languages and prepare it for training.

## Model Architecture

The Transformer model consists of the following components:

- Input Embedding
- Positional Encoding
- Multi-Head Attention
- Feedforward Neural Network
- Layer Normalization
- Encoder and Decoder stacks

Each component is implemented as a separate PyTorch module.

## Training

The model is trained using the following hyperparameters:

- Learning rate: 1e-4
- Batch size: 16
- Number of epochs: 100
- Embedding dimension: 256
- Number of encoder/decoder layers: 6
- Number of attention heads: 8
- Dropout rate: 0.1

The training process includes early stopping and saves the best model based on the lowest loss.

## Evaluation

The model is evaluated using the BLEU (Bilingual Evaluation Understudy) score, a common metric for machine translation tasks.

## Usage

To translate a sentence:

1. Load the trained model
2. Tokenize and encode the input sentence
3. Use the `translate` function to get the translation

Example:

```python
src_sentence = "Hello, how are you?"
src_tokens = ['<sos>'] + word_tokenize(src_sentence) + ['<eos>']
src_ids = [src_token_to_id.get(token, src_token_to_id['<unk>']) for token in src_tokens]
src_tensor = torch.tensor(src_ids).unsqueeze(0).to(device)

translated_ids = translate(transformer_model, src_tensor)
translated_tokens = [tgt_id_to_token[id.item()] for id in translated_ids if id.item() not in [tgt_token_to_id['<sos>'], tgt_token_to_id['<eos>'], tgt_token_to_id['<pad>']]]
translated_sentence = ' '.join(translated_tokens)

print(f"Source: {src_sentence}")
print(f"Translation: {translated_sentence}")
```

For more detailed usage and implementation details, please refer to the Jupyter notebook in this repository.