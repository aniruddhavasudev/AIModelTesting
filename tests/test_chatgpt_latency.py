import time
import torch
from transformers import pipeline
import pytest

# Check if CUDA (GPU) is available and set the device accordingly
device = 0 if torch.cuda.is_available() else -1
if device == 0:
    print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA is not available. Using CPU.")

model_name = "deepseek-ai/deepseek-llm-7b-chat"
# Initialize the text generation pipeline with the desired model
generator = pipeline(
    "text-generation",
    model=model_name,
    device=device,
    torch_dtype=torch.float16 if device == 0 else torch.float32
)

# ✅ Use PyTest parameterization to test multiple prompts
@pytest.mark.parametrize("prompt", [
    "Describe photosynthesis.",
    "Explain how gravity works.",
    "What is artificial intelligence?",
])
def test_generate_text(prompt):
    """Test function to validate text generation latency and throughput."""
    start_time = time.time()
    response = generator(
        prompt,
        max_length=150,
        temperature=0.7,
        do_sample=True
    )
    end_time = time.time()
    generated_text = response[0]["generated_text"]
    latency = round(end_time - start_time, 3)
    tokens = len(generated_text.split())  # Approximate token count
    throughput = round(tokens / latency, 2)  # Tokens per second

    # Print debug info
    print(f"\nPrompt: {prompt}")
    print(f"Generated Text: {generated_text[:100]}...")  # Show first 100 chars
    print(f"Latency: {latency} seconds")
    print(f"Throughput: {throughput} tokens/second")

    # ✅ Add Assertions
    assert latency < 10, "Latency is too high!"
    assert throughput > 1, "Throughput is too low!"
    assert len(generated_text.split()) > 5, "Generated text is too short!"

