import subprocess

if __name__ == "__main__":
    print("Running Bias Tests...")
    subprocess.run(["pytest", "tests/test_chatgpt_bias.py"])

    print("Running Performance Tests...")
    subprocess.run(["pytest", "tests/test_chatgpt_performance.py"])

    print("Running Hugging Face Metrics Tests...")
    subprocess.run(["pytest", "tests/test_chatgpt_hf_metrics.py"])

    print("Running Latency & Throughput Tests...")
    subprocess.run(["pytest", "tests/test_chatgpt_latency.py"])
