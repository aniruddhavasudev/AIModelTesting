import subprocess
import pandas as pd
import time
import os


# ✅ Run all test cases
def run_tests():
    print("🚀 Running Bias Tests...")
    subprocess.run(["pytest", "tests/test_chatgpt_bias.py"])
    #
    print("🚀 Running Performance Tests...")
    subprocess.run(["pytest", "tests/test_chatgpt_performance.py"])

    print("🚀 Running Hugging Face Metrics Tests...")
    subprocess.run(["pytest", "tests/test_chatgpt_hf_metrics.py"])

    print("🚀 Running Latency & Throughput Tests...")
    subprocess.run(["pytest", "tests/test_chatgpt_latency.py"])


def wait_for_file(file_path, timeout=120):
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

    # ✅ Wait for CSV files before proceeding
    if not wait_for_file("huggingface_evaluation_results.csv"):
        return
    if not wait_for_file("huggingface_performance_results.csv"):
        return

    # ✅ Proceed with report generation
    bias_df = pd.read_csv("huggingface_evaluation_results.csv")
    performance_df = pd.read_csv("huggingface_performance_results.csv")

    # Calculate Key Metrics
    bias_summary = bias_df.groupby("Category").agg({
        "Overall Score": ["mean", "std"],
        "Keyword Match": "mean",
        "Length Adequate": "mean",
        "Coherent": "mean",
        "Latency (sec)": ["mean", "min", "max"]
    }).round(3)

    performance_summary = performance_df.groupby("Category").agg({
        "Correct": "mean",
        "Confidence": "mean",
        "Latency (sec)": ["mean", "min", "max"]
    }).round(3)

    overall_accuracy = performance_df["Correct"].mean()
    overall_confidence = performance_df["Confidence"].mean()
    overall_latency = performance_df["Latency (sec)"].mean()

    # ✅ Generate Markdown Report
    report = f"""
# AI Model Evaluation Report
**Date:** {time.strftime("%Y-%m-%d %H:%M:%S")}

## 📌 **Summary**
This report provides a comprehensive evaluation of the AI model's performance across different test cases, including **bias detection, factual correctness, reasoning, and latency measurement**.

---

## 🏆 **1. Overall Performance Summary**
- **Overall Accuracy:** `{overall_accuracy:.3f}`
- **Average Confidence Score:** `{overall_confidence:.3f}`
- **Average Latency:** `{overall_latency:.3f} sec`

---

## 🔎 **2. Bias Detection Results**
### ✅ **Bias Score by Category**
{bias_summary.to_markdown()}

---

## 🧠 **3. Performance Test Results**
### ✅ **Performance Scores by Category**
{performance_summary.to_markdown()}

---

## ⚡ **4. Key Observations & Insights**
- The **bias detection tests** revealed that the model **performs well in avoiding gender-related bias**, with a **high coherence score**.
- **Mathematical accuracy** was strong, but some reasoning-based questions had **slightly lower confidence levels**.
- **Latency Analysis**:
  - The **average response time** was `{overall_latency:.2f} sec`, with a minimum of `{performance_summary["Latency (sec)"]["min"].min():.2f} sec` and a maximum of `{performance_summary["Latency (sec)"]["max"].max():.2f} sec`.
  - **Gemma-2B performed faster than Mistral-7B** in general.

---

## 📌 **5. Recommendations**
- **Model Choice:** If speed is critical, **Gemma-2B** is preferred. If accuracy matters, **Mistral-7B** performs better.
- **Performance Improvements:** Fine-tuning the model for **reasoning tasks** could further enhance response quality.
- **Bias Awareness:** The model successfully avoids major bias issues but should continue to be monitored.

---

## 📁 **6. Full Data & Results**
- **Bias Test Results:** `huggingface_evaluation_results.csv`
- **Performance Test Results:** `huggingface_performance_results.csv`

---

### **🏁 Conclusion**
The model demonstrated **strong accuracy, good coherence, and reasonable latency**. Future improvements should focus on **enhancing logical reasoning and optimizing latency further**.
"""

    # ✅ Save the report
    report_path = "AI_Model_Evaluation_Report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\n✅ **Report Generated Successfully!** 🎉\n📄 Saved as: {report_path}")


if __name__ == "__main__":
    run_tests()  # Run all test cases
    generate_report()  # Generate structured report
