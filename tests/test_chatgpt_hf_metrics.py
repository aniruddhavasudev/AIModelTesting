import openai
import os
import time
import pandas as pd
from tqdm import tqdm

# Load API Key
client = openai.OpenAI(api_key="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")
# Hugging Face LLM-style test prompts
hf_metrics_prompts = {
    "mmlu": [
        "Who discovered gravity?",
        "What is the capital of Canada?",
        "What is the formula for kinetic energy?"
    ],
    "mt_bench": [
        "Translate 'Hello, how are you?' into Spanish.",
        "Solve this riddle: I speak without a mouth and hear without ears. What am I?",
        "Explain the significance of the number Pi."
    ],
    "human_eval": [
        "Write a Python function to sort a list using quicksort.",
        "Implement a SQL query to find duplicate emails in a database.",
        "Generate a simple HTML page with a login form."
    ],
    "chatbot_arena": [
        "You are an AI assistant. A user asks: 'How can I improve my writing skills?'",
        "Someone is struggling with anxiety. Provide a thoughtful response.",
        "A student needs help with calculus. Guide them through differentiation."
    ]
}

# Expected keyword sets
expected_keywords = {
    "mmlu": ["gravity", "Newton", "law", "Canada", "Ottawa", "energy"],
    "mt_bench": ["Spanish", "Pi", "riddle", "answer"],
    "human_eval": ["def", "return", "query", "SELECT", "HTML"],
    "chatbot_arena": ["writing", "improve", "tips", "mental health", "guidance"]
}


# Function to get AI response
def get_ai_response(prompt):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=100
    )
    return response["choices"][0]["message"]["content"].lower()


# Evaluate model across Hugging Face metrics
def evaluate_hf_metrics():
    results = []

    for category, prompts in hf_metrics_prompts.items():
        for prompt in tqdm(prompts, desc=f"Testing {category}"):
            response = get_ai_response(prompt)
            is_correct = any(keyword in response for keyword in expected_keywords[category])

            results.append({
                "Category": category,
                "Prompt": prompt,
                "Response": response,
                "Correct": is_correct
            })

    df = pd.DataFrame(results)
    print(df)


# Run the test
evaluate_hf_metrics()
