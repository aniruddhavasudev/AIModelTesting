import time
import pandas as pd
import torch
from transformers import pipeline

# ✅ Check if CUDA (GPU) is available and set the device accordingly
device = 0 if torch.cuda.is_available() else -1
if device == 0:
    print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA is not available. Using CPU.")

# ✅ Initialize the text generation pipeline with the desired model
generator = pipeline(
    "text-generation",
    model="deepseek-ai/deepseek-llm-7b-chat",
    device=device,
    torch_dtype=torch.float16 if device == 0 else torch.float32
)

# ✅ Define evaluation prompts across different categories
performance_prompts = {
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

# ✅ Expected keywords and evaluation criteria
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
    criteria = expected_keywords.get(category, {})

    # ✅ Handle case where category is missing
    if not criteria:
        return {
            "keyword_match": False,
            "length_adequate": False,
            "coherent": False,
            "overall_score": 0.0
        }

    # ✅ Check for keywords
    keyword_match = any(keyword in response.lower() for keyword in criteria["keywords"])

    # ✅ Check response length
    words = response.split()
    length_adequate = len(words) >= criteria["min_length"]

    # ✅ Check for coherence (basic sentence check)
    sentences = response.split('.')
    coherent = len(sentences) > 1 and all(len(s.strip().split()) > 3 for s in sentences if s.strip())

    return {
        "keyword_match": keyword_match,
        "length_adequate": length_adequate,
        "coherent": coherent,
        "overall_score": round(sum([keyword_match, length_adequate, coherent]) / 3, 2)
    }

def test_performance():
    """Run AI performance testing across all prompt categories."""
    results = []

    print("🚀 Running AI Performance Tests...")

    for category, prompts in performance_prompts.items():
        print(f"\n📌 Testing {category.upper()} category:")

        for prompt in prompts:  # ✅ Fixed: `test_case` is now `prompt`
            response, latency = generate_text(prompt)

            evaluation = evaluate_response(response, category)

            results.append({
                "Category": category,
                "Prompt": prompt,
                "Response": response[:100] + "...",
                "Correct": int(evaluation["keyword_match"]),  # ✅ Ensuring this column exists
                "Confidence": round(evaluation["overall_score"], 2),  # ✅ Ensuring this column exists
                "Latency (sec)": latency if latency else "N/A"
            })

    # ✅ Save Results to CSV
    df = pd.DataFrame(results)

    if not df.empty:
        df.to_csv("/home/rptech/ani_env/AIModelTesting/huggingface_performance_results.csv", index=False)
        print("\n💾 Performance results saved to huggingface_performance_results.csv")
    else:
        print("\n❌ No results generated!")

if __name__ == "__main__":
    test_performance()

