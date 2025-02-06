import requests
import time
import pandas as pd
from tqdm import tqdm
import pytest  # Import the evaluation function

# ✅ Hugging Face API Key (Replace with your actual key)
HF_API_KEY = ""

# ✅ Hugging Face API Endpoint (Use Gemma-2B or Mistral-7B)
API_URL = "https://api-inference.huggingface.co/models/google/gemma-2b"
# Alternative: API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-v0.1"

# ✅ API Headers
HEADERS = {
    "Authorization": f"Bearer {HF_API_KEY}"
}

# ✅ Evaluation Prompts Across Different Categories
evaluation_prompts = {
    "knowledge": [
        "Who discovered gravity?",
        "What is the capital of Canada?",
        "What is the formula for kinetic energy?"
    ],
    "translation": [
        "Translate 'Hello, how are you?' into Spanish.",
        "Solve this riddle: I speak without a mouth and hear without ears. What am I?",
        "Explain the significance of the number Pi."
    ],
    "coding": [
        "Write a Python function to sort a list using quicksort.",
        "Implement a SQL query to find duplicate emails in a database.",
        "Generate a simple HTML page with a login form."
    ],
    "assistance": [
        "You are an AI assistant. A user asks: 'How can I improve my writing skills?'",
        "Someone is struggling with anxiety. Provide a thoughtful response.",
        "A student needs help with calculus. Guide them through differentiation."
    ]
}

# ✅ Expected Keywords and Evaluation Criteria
expected_keywords = {
    "knowledge": {
        "keywords": ["gravity", "newton", "law", "canada", "ottawa", "energy", "kinetic", "mass", "velocity"],
        "min_length": 20
    },
    "translation": {
        "keywords": ["hola", "como", "estás", "eco", "sound", "pi", "3.14", "circle", "mathematics"],
        "min_length": 15
    },
    "coding": {
        "keywords": ["def", "return", "select", "from", "where", "html", "form", "input"],
        "min_length": 30
    },
    "assistance": {
        "keywords": ["practice", "recommend", "suggest", "breathe", "exercise", "derivative", "function"],
        "min_length": 40
    }
}


def query_huggingface(prompt):
    """Send a request to Hugging Face API and retrieve model response."""
    payload = {
        "inputs": prompt,
        "parameters": {"max_length": 150, "temperature": 0.7}
    }

    try:
        start_time = time.time()
        response = requests.post(API_URL, headers=HEADERS, json=payload, timeout=60)
        end_time = time.time()
        response_time = round(end_time - start_time, 3)  # Latency in seconds

        if response.status_code == 200:
            response_data = response.json()
            generated_text = response_data[0]["generated_text"]
            return generated_text.lower(), response_time
        elif response.status_code == 503:
            print("🔄 Model is loading... Retrying in 30 seconds.")
            time.sleep(30)  # Wait & retry
            return query_huggingface(prompt)
        else:
            return f"API Error: {response.status_code} - {response.text}", response_time
    except Exception as e:
        return f"Request Failed: {e}", None


def evaluate_response(response, category):
    """Evaluate AI response based on keyword match, length, and coherence."""
    criteria = expected_keywords[category]

    # Check for keywords
    keyword_match = any(keyword in response for keyword in criteria["keywords"])

    # Check response length
    words = response.split()
    length_adequate = len(words) >= criteria["min_length"]

    # Check for coherence (basic check for sentence structure)
    sentences = response.split('.')
    coherent = len(sentences) > 1 and all(len(s.strip().split()) > 3 for s in sentences if s.strip())

    return {
        "keyword_match": keyword_match,
        "length_adequate": length_adequate,
        "coherent": coherent,
        "overall_score": round(sum([keyword_match, length_adequate, coherent]) / 3, 2)
    }


def run_evaluation():
    """Run AI evaluation across all prompt categories."""
    results = []

    for category, prompts in evaluation_prompts.items():
        print(f"\n📌 Evaluating {category.upper()} category...")

        for prompt in tqdm(prompts, desc=f"🚀 Testing {category}"):
            # Add delay to respect API rate limits
            time.sleep(1)

            response, latency = query_huggingface(prompt)
            evaluation = evaluate_response(response, category)

            results.append({
                "Category": category,
                "Prompt": prompt,
                "Response": response[:100] + "..." if len(response) > 100 else response,  # Truncate long responses
                "Keyword Match": evaluation["keyword_match"],
                "Length Adequate": evaluation["length_adequate"],
                "Coherent": evaluation["coherent"],
                "Overall Score": evaluation["overall_score"],
                "Latency (sec)": latency
            })

    # ✅ Create DataFrame and Calculate Metrics
    df = pd.DataFrame(results)

    # ✅ Print Detailed Results
    print("\n=== 📊 Evaluation Results ===")
    print(df)

    # ✅ Calculate Summary Statistics
    print("\n=== 📈 Summary Statistics ===")
    summary = df.groupby("Category").agg({
        "Overall Score": ["mean", "std"],
        "Keyword Match": "mean",
        "Length Adequate": "mean",
        "Coherent": "mean",
        "Latency (sec)": ["mean", "min", "max"]
    })
    print(summary)

    # ✅ Save Results to CSV
    df.to_csv("huggingface_evaluation_results.csv", index=False)
    print("\n💾 Results saved to huggingface_evaluation_results.csv")

def test_evaluation_runs_without_errors():
    """Ensure the evaluation runs successfully without crashing."""
    try:
        run_evaluation()  # ✅ Call your evaluation function
        assert True  # ✅ If no errors, pass the test
    except Exception as e:
        pytest.fail(f"Evaluation failed due to: {e}")

if __name__ == "__main__":
    test_evaluation_runs_without_errors()
