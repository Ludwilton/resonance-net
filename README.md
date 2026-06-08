# Resonance-net

**Neural network modeling of analog and digital audio equipment**

This repository implements deep learning pipelines for emulating audio equipment such as guitar amplifiers and effect pedals, with a focus on lightweight architectures capable of real-time inference on low-cost hardware.

This code accompanies my thesis, *"Effektiva neurala nätverk för modellering av audioutrustning"* (Efficient Neural Networks for Modeling Audio Equipment), completed as part of the AI and Machine Learning Developer program at ITHS, Gothenburg.

## Overview

Two neural network architectures were implemented and evaluated:

- **LSTM** (Long Short-Term Memory) - Single layer with 32 hidden cells
- **WaveNet / TCN** (Temporal Convolutional Network) - Stacked dilated causal convolutions with gated activations

Both models are designed to capture the non-linear, time-dependent transformations of analog gear while remaining compact enough for real-time inference on resource-constrained hardware.

*This repository contains only the LSTM architecture and training loop used.*
## Results

From the thesis evaluation:

| Aspect | LSTM (32 cells) | WaveNet/TCN |
|--------|----------------|-------------|
| ESR Loss | 0.05-0.07 | 0.05-0.07 |
| Perceptual Quality | Superior  | Acceptable but artificial highs/lows |
| Real-time capability |  Verified (<10ms latency) |  Verified |

!["LSTM results"](/assets/lstm_model_eval.png)
!["TCN results"](/assets/tcn_model_eval.png)

**Successfully modeled:**
- High/low gain amplifiers
- Distortion and fuzz pedals
- IR cab simulation
- Signal chains with input/output EQ
- Various chained combinations of the above

**Limitations identified:**
- Dynamic compression (requires longer temporal context)
- Reverb and delay effects
- Other effects with long time constants

*these were in line with expectations*

## Architecture Details

### LSTM Model
Single-layer LSTM with 32 hidden cells, processing audio in chunks with hidden state passed between buffers for continuous real-time processing. The model predicts a correction signal added to the input via residual connection.

### Loss Function
Error-to-Signal Ratio (ESR) with normalization against target signal energy:
$$L_{\text{ESR}}(\hat{y}, y) = \frac{\sum_{i=1}^{B} \sum_{t=1}^{T} ||e_t^{(i)}||^2}{\sum_{i=1}^{B} \sum_{t=1}^{T} ||y_t^{(i)}||^2}$$

This prevents high-amplitude signals from dominating the loss, unlike standard MSE.

## Dataset Construction

The training dataset that was used combines:
- Direct input (DI) guitar recordings with varied playing techniques
- Frequency sweeps across audible spectrum (0-20,000 Hz)
- Click signals at different frequencies (capturing transients/attack)
- Isolated string pluck recordings
- Various noise types at varying amplitudes

All data at 48kHz sampling rate.

## Repository

`src/rnet_model.py`  LSTM architecture, ESR loss

`src/train.py` Training loop, validation, checkpointing

`src/utils.py` Audio I/O, preprocessing, evaluation utilities

`train_lstm.ipynb` Demo notebook with training & inference examples



## Real-Time Inference

Verified using a C++ implementation with the RTNeural library:
*To be made public, currently hardcoded to my specific hardware*
- Round-trip latency: <10ms on CPU
- Confirms feasibility for low-cost hardware deployment

## Research Basis

This work builds on:
- Juvela et al. (2024) - Single-layer LSTM with 32 hidden cells for audio modeling
- Damskägg, Juvela, Thuillier & Välimäki (2019) - ESR loss function
- Wright, Damskägg & Välimäki (2019) - TCN architectures for audio
- van den Oord et al. (2016) - WaveNet gated activations

## Future Work (Planned)

- Hardware implementation with open-source design
- Support for control variables (gain, EQ knobs) enabling dynamic parameter adjustment



---
For questions about the code or to request the full thesis document, feel free to reach out.

*This project was completed as solo research-based thesis work.*
