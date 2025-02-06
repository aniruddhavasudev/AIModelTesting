import requests
import os
import time
import statistics

# ✅ Hugging Face API Key (Replace with your actual key)
HF_API_KEY = ""

# ✅ Hugging Face API Endpoint (Use Gemma-2B or Mistral-7B)
API_URL = "https://api-inference.huggingface.co/models/google/gemma-2b"
# Alternative: API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-v0.1"

# ✅ API Headers
HEADERS = {
    "Authorization": f"Bearer {HF_API_KEY}"
}

# ✅ Test Prompt for Latency Measurement
TEST_PROMPT = "Describe photosynthesis."


def query_huggingface(prompt):
    """Send a request to Hugging Face API and retrieve model response."""
    payload = {
        "inputs": prompt,
        "parameters": {"max_length": 100, "temperature": 0.7}
    }

    try:
        start_time = time.time()
        response = requests.post(API_URL, headers=HEADERS, json=payload, timeout=60)
        end_time = time.time()
        response_time = round(end_time - start_time, 3)  # Latency in seconds

        if response.status_code == 200:
            response_data = response.json()
            generated_text = response_data[0]["generated_text"]
            tokens = len(generated_text.split())  # Approximate token count
            throughput = round(tokens / response_time, 2)

            return generated_text, response_time, throughput
        elif response.status_code == 503:
            print("🔄 Model is loading... Retrying in 30 seconds.")
            time.sleep(30)  # Wait & retry
            return query_huggingface(prompt)
        else:
            return f"API Error: {response.status_code} - {response.text}", response_time, None
    except Exception as e:
        return f"Request Failed: {e}", None, None


def measure_latency(num_trials=5):
    """Measure API latency and throughput over multiple trials."""
    latencies = []
    throughputs = []

    print(f"🔍 Running {num_trials} latency tests on Hugging Face API...")

    for i in range(num_trials):
        response, latency, throughput = query_huggingface(TEST_PROMPT)

        print(f"\n🧪 Trial {i + 1}:")
        print(f"⏳ Latency: {latency:.2f} sec" if latency else "❌ Latency measurement failed")
        print(f"⚡ Throughput: {throughput:.2f} tokens/sec" if throughput else "❌ Throughput unavailable")
        print(f"💬 Response: {response[:200]}...")  # Show only first 200 characters

        if latency is not None:
            latencies.append(latency)
        if throughput is not None:
            throughputs.append(throughput)

        # Add delay between trials to respect API rate limits
        time.sleep(1)

    # ✅ Display Summary Statistics
    if latencies:
        print("\n=== 📊 Summary Statistics ===")
        print(f"🔄 Number of successful trials: {len(latencies)}")
        print(f"⏱️  Average latency: {statistics.mean(latencies):.2f} sec")
        print(f"📊 Latency std dev: {statistics.stdev(latencies) if len(latencies) > 1 else 0:.2f} sec")
        print(f"⚡ Average throughput: {statistics.mean(throughputs):.2f} tokens/sec")
        print(f"⬇️  Min latency: {min(latencies):.2f} sec")
        print(f"⬆️  Max latency: {max(latencies):.2f} sec")


if __name__ == "__main__":
    measure_latency()
