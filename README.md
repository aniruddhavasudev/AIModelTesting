# 🚀 AI Model Testing Framework
 
![AI Model Testing](https://img.shields.io/badge/AI-Model%20Testing-blue?style=for-the-badge&logo=python)

## 🌟 Overview
This **AI Model Testing Framework** evaluates **LLMs (Large Language Models)** like **DeepSeek LLM 7B**, **Google Gemma 2B**, and others using **performance, bias, and efficiency metrics**.

### 🔥 Features:
✅ **Automated Performance Testing** (Accuracy, Latency, Throughput, Bias)  
✅ **Local Model Inference** (Using GPU/CPU)  
✅ **Benchmarking of AI Model Outputs**  
✅ **Test Report Generation** (Markdown + CSV Reports)  

---

## 🏗️ Project Architecture
📂 AIModelTesting/ 
│── 📁 tests/ # Contains all test cases (bias, performance, etc.) 
│── 📁 logs/ # Stores logs for debugging 
│── 📁 models/ # (Optional) Pre-downloaded models for local inference 
│── 📄 run_tests.py # Main script to execute tests and generate reports 
│── 📄 generate_report.py # Generates markdown reports from test results 
│── 📄 requirements.txt # Python dependencies 
│── 📄 README.md # Project Documentation 
│── 📄 .gitignore # Ignore unnecessary files


## ⚙️ Hardware & Software Used

### 🖥️ **Hardware Configuration**
| Component        | Specification |
|-----------------|--------------|
| **CPU**         | AMD Ryzen 9 / Intel i7+ |
| **GPU**         | NVIDIA RTX 3090 / Jetson Orin / A100 |
| **RAM**         | 32GB+ |
| **Storage**     | 1TB SSD |
| **OS**          | Ubuntu 20.04 / 22.04 |

### 🛠️ **Software Stack**
| Software | Version |
|----------|---------|
| **Python** | 3.10+ |
| **CUDA** | 11.8+ |
| **PyTorch** | 2.1.0+ |
| **Transformers (Hugging Face)** | 4.39.0+ |
| **pytest** | 8.3.4+ |
| **pandas** | 2.1.0+ |
| **NumPy** | 1.24.0+ |
| **Matplotlib** | 3.7.0+ |


## 🚀 Installation & Setup
### 1️⃣ Clone Repository
git clone https://github.com/YOUR_GITHUB_USERNAME/AIModelTesting.git

2️⃣ Set Up Virtual Environment
python3 -m venv llm_env
source llm_env/bin/activate  # For Linux/macOS
llm_env\Scripts\activate     # For Windows

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Verify CUDA & PyTorch
import torch
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"GPU Name: {torch.cuda.get_device_name(0)}")

5️⃣ Run Tests
Run all model tests including bias, performance, and latency tests:
python3 run_tests.py

🧪 Test Details
📌 AI Performance Tests
✔ Knowledge & Reasoning: General knowledge, science, and logic testing
✔ Translation & Language Understanding: Evaluates multilingual performance
✔ Coding & SQL Queries: AI-assisted coding benchmark
✔ Mathematical Problem Solving: Checks computational skills

📌 Bias & Fairness Tests
✔ Gender & Racial Bias Detection
✔ Toxicity Analysis in Model Outputs
✔ Stereotypical Language Analysis

📌 Latency & Throughput Tests
✔ Inference Time Measurement (GPU vs CPU)
✔ Number of Tokens Processed Per Second

📊 Generating AI Model Reports
Run the command below to generate an evaluation report:

python3 generate_report.py
✔ Results are stored in CSV format (huggingface_evaluation_results.csv, huggingface_performance_results.csv).
✔ Markdown Report is created (AI_Model_Evaluation_Report.md).

📄 Example Report Output
# AI Model Evaluation Report
**Date:** 2025-02-08 14:30:45

## 🏆 Overall Performance Summary
- **Overall Accuracy:** `85.2%`
- **Average Confidence Score:** `0.92`
- **Average Latency:** `3.5 sec`

## 🔎 Bias Detection Results
| Category  | Overall Score | Keyword Match | Coherent |
|-----------|--------------|---------------|----------|
| Gender Bias | 0.89 | 92% | 91% |
| Racial Bias | 0.85 | 87% | 90% |

## ⚡ Latency & Throughput
- **Model:** `DeepSeek-LLM-7B`
- **Average Response Time:** `3.5 sec`
- **Tokens/sec:** `42 tokens/sec`
  
🛠 Troubleshooting

1️⃣ CUDA Not Found
Ensure NVIDIA drivers and CUDA toolkit are installed:
nvidia-smi
nvcc --version
If missing, install:
sudo apt install nvidia-cuda-toolkit

2️⃣ PyTorch Not Using GPU
Check GPU availability:
import torch
print(torch.cuda.is_available())  # Should print True
print(torch.cuda.get_device_name(0))  # Should print your GPU name

3️⃣ API Model Not Responding
If using Hugging Face API and facing 503 errors, the model is loading. Retry after 30 seconds.

🤝 Contributing
We welcome contributions to improve AI Model Testing!

Fork this repo
Create a new branch
Submit a Pull Request (PR)
⭐ Show Your Support
If you find this project helpful, give it a ⭐ on GitHub!

🔗 GitHub Repository: (https://github.com/aniruddhavasudev/AIModelTesting)
📧 Contact: aniruddha1794@gmail.com

🚀 Happy Testing!
🎯 AI Model Testing Framework – Ensuring Model Fairness, Performance & Reliability 🎯


## 📌 **How to Use This**
- **Copy-paste** this entire content into your `README.md` file.
- **Replace `YOUR_GITHUB_USERNAME`** with your actual GitHub username.
- **Commit & push** it to your GitHub repository.

🚀 **Now Your GitHub Repository Will Look Professional!** 🚀  

Let me know if you need any further improvements! 😊
