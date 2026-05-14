# Multimodal Aggression Detection: ViViT Temporal Model

## Overview
This repository contains the temporal action-recognition component of our Dual-Stream Aggression Detection System. We utilize a Vision Transformer for Video Classification (ViViT) to process overlapping sliding windows of video frames. This model is designed to recognize macro-motions indicative of violence (e.g., fighting, striking) while mitigating the temporal dilution of brief violent events across long video sequences using LogSumExp (Softmax) pooling.

## Dataset
The model is trained and evaluated on the Real Life Violence Situations Dataset.
* **Dataset Link:** [Kaggle: Real Life Violence Situations Dataset](https://www.kaggle.com/datasets/mohamedmustafa/real-life-violence-situations-dataset)

## Code Organization
The repository is structured to separate execution environments from source code:


* `run.py`: The deployment script that configures and launches the training job on AWS SageMaker using an `ml.g5.12xlarge` instance. 
Will require separate AWS credentials and resources to run.
* `requirements.txt`: Contains all necessary Python packages and specific versions required to run the environment.
* `src/`: The core source code directory.
  * `model.py`: Defines and returns modified ViVit model
  * `dataset.py`: Defines the `ViolentVideoDataset` class. Handles loading pre-processed `.pt` tensor files to eliminate video I/O bottlenecks and applies spatial/temporal augmentations.
  * `train.py`: Contains the primary training loop, validation logic, and testing metrics. Implements Distributed Data Parallel (DDP), Automatic Mixed Precision (AMP), and sliding window temporal extraction. 
  Run this to run the full training on model. Will default to cpu if cuda is not available.

## Requirements and Installation
This project was built using **Python 3.11** and PyTorch. 

To set up the environment locally, clone the repository and install the dependencies:

```bash
git clone [https://github.com/your-org/ViViT_Model.lang](https://github.com/your-org/ViViT_Model.lang)
cd ViViT_Model
pip install -r requirements.txt
