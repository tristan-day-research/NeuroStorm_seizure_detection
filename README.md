# NeuroStorm_seizure_detection
This is a Large Brain Model (LBM), with a similar architecture of an LLM. NeuroStorm detects what type of seizure, if any, is present in an EEG brain signal recording. For the tokenizer it uses a Variational Autoencoder based on Fast Fourier Transforms of EEG signal patches. These tokens are used to train a transformer model.

This work is a replication of https://openreview.net/pdf?id=QzTpTRVtrP specifically tailored to detect seizures. 
