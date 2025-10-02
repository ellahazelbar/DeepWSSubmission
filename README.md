# ASL Translator (Single-Sign Recognition)

This project implements a deep learning system to translate single American Sign Language (ASL) signs into English words using video input. It leverages MediaPipe for pose and hand landmark extraction and a BiLSTM network for temporal sequence modeling.

---

## Overview

Sign languages are a primary mode of communication for deaf and hard-of-hearing individuals, but barriers still exist between signers and non-signers. This project focuses on recognizing isolated ASL signs from short video clips and translating them into English words. The approach can serve as a foundation for larger ASL-to-English translation systems.

---

## Features

- Recognizes single ASL signs from video input.  
- Uses MediaPipe Holistic for hand and upper-body landmark extraction.  
- Employs BiLSTM for modeling temporal dynamics of the gestures.  
- Written in PyTorch for flexibility and GPU acceleration.  
- Easily extendable to more signs with additional data.

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/ellahazelbar/DeepWSSubmission.git
cd asl-translator
```

2. Install Dependencies
```bash
pip install -r requirements.txt
```

## Usage
1. Offline Usage - Interpret a prerecorded video file
```bash
python.exe ./demo/predict.py <input_video>
```

2. Realtime Usage - record a video using the device camera and interpret it
```bash
python.exe ./demo/predict_realtime.py
```
