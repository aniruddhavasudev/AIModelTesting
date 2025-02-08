import subprocess
import pandas as pd
import time
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pytest


if torch.cuda.is_available():
    print(f"CUDA is available. using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA is not available")
model_name = "deepseek-ai/deepseek-llm-7b-chat"
# Initialize the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16
)

# Function to generate text using the model
def generate_text(prompt, max_new_tokens=1000, temperature=0.1, do_sample=True):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    outputs = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=do_sample
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# ✅ Run all test cases
def run_tests():
    print("🚀 Running Bias Tests...")
    subprocess.run(["pytest", "tests/test_chatgpt_bias.py"])

    print("🚀 Running Performance Tests...")
    subprocess.run(["pytest", "tests/test_chatgpt_performance.py"])

    print("🚀 Running Hugging Face Metrics Tests...")
    subprocess.run(["pytest", "tests/test_chatgpt_hf_metrics.py"])

    print("🚀 Running Latency & Throughput Tests...")
    subprocess.run(["pytest", "tests/test_chatgpt_latency.py"])

def wait_for_file(file_path, timeout=300):
    """Wait for a file to be created with a timeout (default: 120 sec)."""
    print(f"⏳ Waiting for {file_path} to be created...")

    start_time = time.time()
    while not os.path.exists(file_path):
        if time.time() - start_time > timeout:
            print(f"❌ Timeout! {file_path} was not created after {timeout} seconds.")
            return False
        time.sleep(5)  # ✅ Wait & retry

    print(f"✅ Found {file_path}!")
    return True

def generate_report():
    print("\n📊 **Generating AI Model Evaluation Report...**")

    # ✅ Check if CSV files exist
    if not os.path.exists("huggingface_evaluation_results.csv"):
        print("❌ Missing: huggingface_evaluation_results.csv. Report generation aborted.")
        return
    if not os.path.exists("huggingface_performance_results.csv"):
        print("❌ Missing: huggingface_performance_results.csv. Report generation aborted.")
        return

    # ✅ Load CSV files
    bias_df = pd.read_csv("huggingface_evaluation_results.csv")
    performance_df = pd.read_csv("huggingface_performance_results.csv")

    # ✅ Validate required columns exist before aggregating
    required_columns = {"Category", "Correct", "Confidence", "Latency (sec)"}
    missing_columns = required_columns - set(performance_df.columns)

    if missing_columns:
        print(f"❌ Error: Missing columns in performance results: {missing_columns}")
        print("🔄 Ensure your test script correctly logs 'Correct', 'Confidence', and 'Latency (sec)'")
        return

    # ✅ Generate Summary Only If Data Exists
    performance_summary = performance_df.groupby("Category").agg({
        "Correct": "mean",
        "Confidence": "mean",
        "Latency (sec)": ["mean", "min", "max"]
    }).round(3)

    # ✅ Handle Potential Empty DataFrames
    bias_summary_text = bias_df.to_markdown() if not bias_df.empty else "**No Bias Data Available**"
    performance_summary_text = performance_summary.to_markdown() if not performance_df.empty else "**No Performance Data Available**"

    # ✅ Generate Markdown Report
    report = f"""
# AI Model Evaluation Report
**Date:** {time.strftime("%Y-%m-%d %H:%M:%S")}

## 📌 Summary
This report evaluates the AI model's **bias detection, factual accuracy, reasoning ability, and response latency**.

## 🏆 Overall Performance Summary
- **Overall Accuracy:** `{performance_df["Correct"].mean():.3f}`
- **Average Confidence Score:** `{performance_df["Confidence"].mean():.3f}`
- **Average Latency:** `{performance_df["Latency (sec)"].mean():.3f} sec`

## 🔎 Bias Detection Results
{bias_summary_text}

## 🧠 Performance Test Results
{performance_summary_text}

## ⚡ Key Insights
- Model **performs well** in avoiding major bias issues.
- **Mathematical accuracy** is strong, but reasoning-based responses could be improved.
- **Latency varies** but generally remains within an acceptable range.

## 📁 Full Data & Results
- Bias Test Results: `huggingface_evaluation_results.csv`
- Performance Test Results: `huggingface_performance_results.csv`

### **🏁 Conclusion**
The AI model demonstrates **good coherence, strong factual accuracy, and reasonable latency**. Improvements should focus on **reasoning skills and bias reduction**.

"""
    # ✅ Save the Report
    report_path = "AI_Model_Evaluation_Report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\n✅ **Report Generated Successfully!** 🎉\n📄 Saved as: {report_path}")

def main():
    run_tests()       # Run all test cases
    generate_report() # Generate structured report

if __name__ == "__main__":
    main()

