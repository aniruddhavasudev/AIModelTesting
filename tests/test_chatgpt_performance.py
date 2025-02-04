import openai
import os
import pandas as pd
from sklearn.metrics import accuracy_score

# Load API Key
client = openai.OpenAI(api_key="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")

performance_prompts = {
    "math": "What is the square root of 144?",
    "reasoning": "Explain why the sky is blue."
}

expected_answers = {
    "math": "12",
    "reasoning": ["scattering", "light", "atmosphere"]
}

def get_ai_response(prompt):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=100
    )
    return response.choices[0].message.content.lower()

def test_performance():
    y_true, y_pred = [], []
    for category, prompt in performance_prompts.items():
        response = get_ai_response(prompt)
        correct = any(ans in response for ans in expected_answers[category])
        y_true.append(1)
        y_pred.append(1 if correct else 0)

    accuracy = accuracy_score(y_true, y_pred)
    print(f"✅ Model Accuracy: {accuracy:.2f}")

test_performance()
