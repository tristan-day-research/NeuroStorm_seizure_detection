**NeuroStorm** is a Large Brain Model (LaBraM), inspired by the architecture of Large Language Models (LLMs).
This project focuses on detecting and classifying seizures from EEG brain signal recordings. It aims to assist neurologists by automatically detecting anomalous neural activity, helping to reduce diagnostic workload.

## How It Works:
- Tokenizer – NeuroStorm uses a Variational Autoencoder (VAE) to tokenize EEG signals. The VAE processes Fast Fourier Transforms (FFT) of EEG signal patches, generating compact representations of brain activity.
- Transformer Model – These tokens are used to train a transformer, similar to those found in natural language processing, but adapted for neurodata.
- Classifier – The transformer's output is passed to a classifier that identifies the type of seizure (if present).

## Why "NeuroStorm"?
Seizures are often described as electrical storms in the brain. NeuroStorm reflects this analogy.

## Project Foundation:
This project replicates and extends the work presented in the paper:
"LARGE BRAIN MODEL FOR LEARNING GENERIC REPRESENTATIONS WITH TREMENDOUS EEG DATA IN BCI"
Read the paper here https://openreview.net/pdf?id=QzTpTRVtrP
Authors: Wei-Bang Jiang, Li-Ming Zhao, and Bao-Liang Lu at Shanghai Jiao Tong University and Shanghai Emotionhelper Technology Co., Ltd.
NeuroStorm adapts LaBraM specifically for seizure detection, building on their groundbreaking research in Brain-Computer Interfaces (BCI).
