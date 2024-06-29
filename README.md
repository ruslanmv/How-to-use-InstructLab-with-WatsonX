# How to Use InstructLab with Watsonx.ai

## Introduction

In the rapidly evolving field of artificial intelligence, the need for efficient and effective methods to fine-tune large language models (LLMs) is ever-present. InstructLab provides a model-agnostic, open-source platform that facilitates contributions to LLMs through community-driven efforts. This approach enables the creation of robust models tailored to specific knowledge domains using synthetic-data-based alignment tuning.

In this tutorial, we will guide you through the process of setting up InstructLab on a Linux environment using Windows Subsystem for Linux (WSL) on Ubuntu 22.04. Following the installation, we will demonstrate how to fine-tune an LLM to create a medical chatbot, leveraging the dataset and methods we discussed earlier. By the end of this guide, you will have a comprehensive understanding of how to utilize InstructLab for your AI projects.

## Prerequisites

- **Windows 10 or 11** with WSL installed
- **Ubuntu 22.04** setup in WSL
- Basic understanding of Linux commands and Python
- Internet connection for downloading necessary tools and datasets

## Step-by-Step Installation of InstructLab on Ubuntu 22.04 (WSL)

### Step 1: Set Up WSL and Install Ubuntu 22.04

1. **Enable WSL**:
    ```powershell
    wsl --install
    ```

2. **Install Ubuntu 22.04**:
    - Open the Microsoft Store, search for "Ubuntu 22.04", and click "Install".

3. **Launch Ubuntu**:
    - Open Ubuntu from the Start menu and follow the initial setup instructions.

### Step 2: Install Necessary Dependencies

1. **Update Package List**:
    ```bash
    sudo apt update
    ```

2. **Install Python and Pip**:
    ```bash
    sudo apt install python3 python3-pip -y
    ```

3. **Install Git**:
    ```bash
    sudo apt install git -y
    ```

4. **Install Other Dependencies**:
    ```bash
    sudo apt install build-essential libssl-dev libffi-dev -y
    ```

### Step 3: Install InstructLab CLI

1. **Clone the InstructLab Repository**:
    ```bash
    git clone https://github.com/InstructLab/instructlab-cli.git
    cd instructlab-cli
    ```

2. **Install InstructLab CLI**:
    ```bash
    pip install .
    ```

3. **Initialize InstructLab**:
    ```bash
    ilab init --non-interactive
    ```

### Step 4: Download the Base Model

1. **Download the Default Model**:
    ```bash
    ilab download
    ```

2. **Optional: Download a Specific Model**:
    ```bash
    ilab download --repository <huggingface_repo> --filename <filename>.gguf
    ```

## Creating and Training a Medical Chatbot with InstructLab

### Step 1: Prepare the Dataset

1. **Load and Clean the Dataset**:
    ```python
    import pandas as pd
    import re
    from datasets import load_dataset

    dataset_ = load_dataset("ruslanmv/ai-medical-chatbot")
    train_data = dataset_["train"]
    df = pd.DataFrame(train_data)
    df = df[["Description", "Doctor"]].rename(columns={"Description": "question", "Doctor": "answer"})
    
    df['question'] = df['question'].apply(lambda x: re.sub(r'\s+', ' ', x.strip()))
    df['answer'] = df['answer'].apply(lambda x: re.sub(r'\s+', ' ', x.strip()))
    df['question'] = df['question'].str.lstrip('Q. ')
    df['answer'] = df['answer'].str.replace('-->', '')
    ```

2. **Split the Dataset**:
    ```python
    num = 1000  # Example: Number of training samples
    df_train = df.iloc[:num, :]
    df_test = df.iloc[num:num+100, :]
    df_train.to_csv('train.csv', index=False)
    df_test.to_csv('test.csv', index=False)
    ```

### Step 2: Create Required Files for Training

1. **Create `qna.yaml`**:
    ```yaml
    task_description: Train a medical chatbot
    created_by: YourGithubUsername
    domain: healthcare
    seed_examples:
    - question: What does abutment of the nerve root mean?
      answer: Hi. I have gone through your query with diligence...
    - question: What should I do to reduce my weight gained during pregnancy?
      answer: Hi. You have really done well with the hypothy...
    document:
      repo: https://github.com/YourGithubUsername/YourRepo
      commit: <commit_SHA>
      patterns:
      - *.md
    ```

2. **Create `attribution.txt`**:
    ```text
    [Link to source]
    [Link to work]
    [License of the work]
    [Creator name]
    ```

### Step 3: Generate Synthetic Data

1. **Generate Data**:
    ```bash
    ilab generate
    ```

2. **Generate Additional Samples**:
    ```bash
    ilab generate --num-instructions 200
    ```

### Step 4: Train the Model

1. **Train the Model**:
    ```bash
    ilab train
    ```

2. **Optional: Train on GPU**:
    ```bash
    ilab train --device 'cuda'
    ```

### Step 5: Test the Model

1. **Test the Model**:
    ```bash
    ilab test
    ```

2. **Optional: Convert Model for MacOS**:
    ```bash
    ilab convert
    ```

### Step 6: Serve and Chat with the Model

1. **Serve the Model**:
    ```bash
    ilab serve --model-path instructlab-granite-7b-lab-trained/instructlab-granite-7b-lab-Q4_K_M.gguf
    ```

2. **Chat with the Model**:
    ```bash
    ilab chat -gm -m instructlab-granite-7b-lab-trained/instructlab-granite-7b-lab-Q4_K_M.gguf
    ```

### Summary

In this comprehensive guide, we have walked through the steps to set up InstructLab on a WSL Ubuntu 22.04 environment, prepare a dataset, and fine-tune a large language model to create a medical chatbot. This tutorial showcased the installation process, dataset preparation, synthetic data generation, model training, and testing, culminating in deploying and interacting with the chatbot.

By following this guide, you can leverage InstructLab to develop customized AI solutions for various domains. Join the InstructLab community on GitHub and start contributing to the open-source LLM landscape.