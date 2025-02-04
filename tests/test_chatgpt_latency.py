import openai
import os
import time

# Load API Key
client = openai.OpenAI(api_key="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")


def measure_latency():
    start_time = time.time()

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Describe photosynthesis."}],
        max_tokens=100
    )

    end_time = time.time()
    response_time = end_time - start_time
    tokens = len(response.choices[0].message.content.split())
    throughput = tokens / response_time

    print(f"⏳ Latency: {response_time:.2f} sec")
    print(f"⚡ Throughput: {throughput:.2f} tokens/sec")


measure_latency()
