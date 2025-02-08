import time
import pandas as pd
import torch
from transformers import pipeline
from tqdm import tqdm
import pytest


# Check if CUDA (GPU) is available and set the device
device = 0 if torch.cuda.is_available() else -1
if device == 0:
    print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA is not available. Using CPU.")

# Initialize the text generation pipeline with the desired model
model_name = "deepseek-ai/deepseek-llm-7b-chat"  # Replace with your model
generator = pipeline(
    "text-generation",
    model=model_name,
    device=device,
    torch_dtype=torch.float16 if device == 0 else torch.float32
)

# Evaluation prompts across different categories
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

# Expected keywords and evaluation criteria
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

def generate_text(prompt, max_length=150, temperature=0.7):
    """Generate text using the local model."""
    start_time = time.time()
    response = generator(
        prompt,
        max_length=max_length,
        temperature=temperature,
        do_sample=True
    )
    end_time = time.time()
    generated_text = response[0]["generated_text"]
    latency = round(end_time - start_time, 3)
    return generated_text, latency

def evaluate_response(response, category):
    """Evaluate AI response based on keyword match, length, and coherence."""
    criteria = expected_keywords[category]

    # Check for keywords
    keyword_match = any(keyword in response.lower() for keyword in criteria["keywords"])

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

def test_run_evaluation():
    """Run AI evaluation across all prompt categories."""
    results = []

    for category, prompts in evaluation_prompts.items():
        print(f"\n📌 Evaluating {category.upper()} category...")

        for prompt in tqdm(prompts, desc=f"🚀 Testing {category}"):
            # Add delay to prevent potential issues with rapid requests
            time.sleep(1)

            response, latency = generate_text(prompt)
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

    # Create DataFrame and Calculate Metrics
    df = pd.DataFrame(results)

    # Print Detailed Results
    print("\n=== 📊 Evaluation Results ===")
    print(df)

    # Calculate Summary Statistics
    print("\n=== 📈 Summary Statistics ===")
    summary = df.groupby("Category").agg({
        "Overall Score": ["mean", "std"],
        "Keyword Match": "mean",
        "Length Adequate": "mean",
        "Coherent": "mean",
        "Latency (sec)": ["mean", "min", "max"]
    })
    print(summary)

    # Save Results to CSV
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    csv_filename = f"/home/rptech/ani_env/AIModelTesting/huggingface_evaluation_results.csv"
    df.to_csv(csv_filename, index=False)
    print(f"\n💾 Results saved to {csv_filename}")

if __name__ == "__main__":
    test_run_evaluation()

