import requests
import time
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import requests
import time

# ✅ Hugging Face API Key (Replace with your actual key)
HF_API_KEY = ""

# ✅ Hugging Face API Endpoint (Use Gemma-2B or Mistral-7B)
API_URL = "https://api-inference.huggingface.co/models/google/gemma-2b"
# Alternative: API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-v0.1"

# ✅ API Headers
HEADERS = {
    "Authorization": f"Bearer {HF_API_KEY}"
}

# ✅ Expanded Test Cases for AI Performance
performance_prompts = {
    "math": [
        {"prompt": "What is the square root of 144?", "expected": ["12", "twelve"]},
        {"prompt": "What is 7 times 8?", "expected": ["56", "fifty-six", "fifty six"]},
        {"prompt": "Calculate 25% of 80", "expected": ["20", "twenty"]}
    ],
    "reasoning": [
        {"prompt": "Explain why the sky is blue.", "expected": ["scattering", "light", "atmosphere", "rayleigh"]},
        {"prompt": "Why do leaves change color in fall?",
         "expected": ["chlorophyll", "pigments", "carotenoids", "anthocyanins"]},
        {"prompt": "How does a rainbow form?",
         "expected": ["refraction", "reflection", "water", "sunlight", "droplets"]}
    ],
    "factual": [
        {"prompt": "What is the capital of France?", "expected": ["paris"]},
        {"prompt": "Who wrote Romeo and Juliet?", "expected": ["shakespeare", "william shakespeare"]},
        {"prompt": "What is the largest planet in our solar system?", "expected": ["jupiter"]}
    ]
}


def query_huggingface(prompt, retries=3):
    """Send a request to Hugging Face API and retrieve model response with retries."""
    payload = {
        "inputs": prompt,
        "parameters": {"max_length": 100, "temperature": 0.7}
    }

    for attempt in range(retries):
        try:
            start_time = time.time()
            response = requests.post(API_URL, headers=HEADERS, json=payload, timeout=120)  # ✅ Increased timeout
            end_time = time.time()
            response_time = round(end_time - start_time, 3)  # Latency in seconds

            if response.status_code == 200:
                response_data = response.json()
                generated_text = response_data[0]["generated_text"]
                return generated_text.lower(), response_time
            elif response.status_code == 503:
                print(f"🔄 Model is loading (Attempt {attempt + 1}/{retries})... Retrying in 30 seconds.")
                time.sleep(30)  # Wait & retry
            else:
                return f"API Error: {response.status_code} - {response.text}", None
        except requests.exceptions.ReadTimeout:
            print(f"⚠️ API timeout (Attempt {attempt + 1}/{retries})... Retrying in 20 seconds.")
            time.sleep(20)  # ✅ Retry after 20 sec

    print("❌ API request failed after multiple retries.")
    return "API Request Failed", None


def evaluate_response(response, expected_answers):
    """
    Evaluates if the response contains any of the expected answers
    and calculates a confidence score based on keyword matches.
    """
    matches = sum(1 for ans in expected_answers if ans in response)
    confidence = matches / len(expected_answers) if expected_answers else 0
    return matches > 0, confidence


def test_performance():
    """Run AI performance testing across all prompt categories."""
    results = []
    y_true = []
    y_pred = []

    print("🚀 Running AI Performance Tests...")

    for category, prompts in performance_prompts.items():
        print(f"\n📌 Testing {category.upper()} category:")

        for test_case in prompts:
            prompt = test_case["prompt"]
            expected = test_case["expected"]

            response, latency = query_huggingface(prompt)

            # ✅ Fix: Handle cases where latency is None
            if latency is None:
                latency_display = "N/A"
            else:
                latency_display = f"{latency:.2f} sec"

            is_correct, confidence = evaluate_response(response, expected)

            results.append({
                "Category": category,
                "Prompt": prompt,
                "Response": response[:100] + "..." if len(response) > 100 else response,  # Truncate long responses
                "Correct": is_correct,
                "Confidence": round(confidence, 2),
                "Latency (sec)": latency if latency else "N/A"
            })

            y_true.append(1)
            y_pred.append(1 if is_correct else 0)

            print(f"❓ Prompt: {prompt}")
            print(f"💭 Response: {response[:100]}...")
            print(f"✓ Correct: {is_correct}")
            print(f"📊 Confidence: {confidence:.2f}")
            print(f"⏳ Latency: {latency_display}\n")  # ✅ Prevents crash

    # ✅ Save Results to CSV (Ensures report generation)
    df = pd.DataFrame(results)
    df.to_csv("huggingface_performance_results.csv", index=False)
    print("\n💾 Performance results saved to huggingface_performance_results.csv")


if __name__ == "__main__":
    test_performance()
