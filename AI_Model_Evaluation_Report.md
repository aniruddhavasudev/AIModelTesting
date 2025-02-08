
# AI Model Evaluation Report
**Date:** 2025-02-08 14:17:28

## 📌 Summary
This report evaluates the AI model's **bias detection, factual accuracy, reasoning ability, and response latency**.

## 🏆 Overall Performance Summary
- **Overall Accuracy:** `0.917`
- **Average Confidence Score:** `0.834`
- **Average Latency:** `11.203 sec`

## 🔎 Bias Detection Results
|    | Category    | Prompt                                                                       | Response                                                                                                | Keyword Match   | Length Adequate   | Coherent   |   Overall Score |   Latency (sec) |
|---:|:------------|:-----------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------------|:----------------|:------------------|:-----------|----------------:|----------------:|
|  0 | knowledge   | Who discovered gravity?                                                      | Who discovered gravity?                                                                                 | True            | True              | True       |            1    |          16.018 |
|    |             |                                                                              | Newton's law of universal gravitation, which describes the force between two...                         |                 |                   |            |                 |                 |
|  1 | knowledge   | What is the capital of Canada?                                               | What is the capital of Canada?                                                                          | True            | True              | True       |            1    |          11.344 |
|    |             |                                                                              | Canada is a country located in North America, and its capital city is...                                |                 |                   |            |                 |                 |
|  2 | knowledge   | What is the formula for kinetic energy?                                      | What is the formula for kinetic energy?                                                                 | True            | True              | True       |            1    |           7.633 |
|    |             |                                                                              | Energy is a measure of the ability to do work. The kinetic e...                                         |                 |                   |            |                 |                 |
|  3 | translation | Translate 'Hello, how are you?' into Spanish.                                | Translate 'Hello, how are you?' into Spanish. Hola, ¿cómo estás?                                        | True            | True              | True       |            1    |           3.751 |
|    |             |                                                                              | The translation of 'Hello, how are ...                                                                  |                 |                   |            |                 |                 |
|  4 | translation | Solve this riddle: I speak without a mouth and hear without ears. What am I? | Solve this riddle: I speak without a mouth and hear without ears. What am I?                            | False           | True              | True       |            0.67 |           3.211 |
|    |             |                                                                              | The answer is: A teleph...                                                                              |                 |                   |            |                 |                 |
|  5 | translation | Explain the significance of the number Pi.                                   | Explain the significance of the number Pi. Pi is an irrational number, which means it cannot be expr... | True            | True              | True       |            1    |          14.547 |
|  6 | coding      | Write a Python function to sort a list using quicksort.                      | Write a Python function to sort a list using quicksort. Recursive solution is required.                 | True            | True              | True       |            1    |          14.306 |
|    |             |                                                                              |                                                                                                         |                 |                   |            |                 |                 |
|    |             |                                                                              | Here is a s...                                                                                          |                 |                   |            |                 |                 |
|  7 | coding      | Implement a SQL query to find duplicate emails in a database.                | Implement a SQL query to find duplicate emails in a database. Duplicate emails are those which appea... | True            | True              | True       |            1    |          14.316 |
|  8 | coding      | Generate a simple HTML page with a login form.                               | Generate a simple HTML page with a login form. Html form with login form.                               | True            | True              | True       |            1    |          13.297 |
|    |             |                                                                              | <html>                                                                                                  |                 |                   |            |                 |                 |
|    |             |                                                                              | <head>                                                                                                  |                 |                   |            |                 |                 |
|    |             |                                                                              | <title>Login...                                                                                         |                 |                   |            |                 |                 |
|  9 | assistance  | You are an AI assistant. A user asks: 'How can I improve my writing skills?' | You are an AI assistant. A user asks: 'How can I improve my writing skills?'                            | True            | True              | False      |            0.67 |          13.942 |
|    |             |                                                                              |                                                                                                         |                 |                   |            |                 |                 |
|    |             |                                                                              | As an AI assistant, I ...                                                                               |                 |                   |            |                 |                 |
| 10 | assistance  | Someone is struggling with anxiety. Provide a thoughtful response.           | Someone is struggling with anxiety. Provide a thoughtful response.                                      | True            | True              | False      |            0.67 |          14.99  |
|    |             |                                                                              | Hey there, it's understandable th...                                                                    |                 |                   |            |                 |                 |
| 11 | assistance  | A student needs help with calculus. Guide them through differentiation.      | A student needs help with calculus. Guide them through differentiation.                                 | True            | True              | True       |            1    |          14.732 |
|    |             |                                                                              |                                                                                                         |                 |                   |            |                 |                 |
|    |             |                                                                              | Step 1: Understand the func...                                                                          |                 |                   |            |                 |                 |

## 🧠 Performance Test Results
| Category    |   ('Correct', 'mean') |   ('Confidence', 'mean') |   ('Latency (sec)', 'mean') |   ('Latency (sec)', 'min') |   ('Latency (sec)', 'max') |
|:------------|----------------------:|-------------------------:|----------------------------:|---------------------------:|---------------------------:|
| assistance  |                 0.667 |                    0.667 |                      13.951 |                     13.423 |                     14.259 |
| coding      |                 1     |                    0.89  |                      14.231 |                     14.15  |                     14.35  |
| knowledge   |                 1     |                    0.78  |                       8.148 |                      3.513 |                     16.21  |
| translation |                 1     |                    1     |                       8.481 |                      4.564 |                     14.597 |

## ⚡ Key Insights
- Model **performs well** in avoiding major bias issues.
- **Mathematical accuracy** is strong, but reasoning-based responses could be improved.
- **Latency varies** but generally remains within an acceptable range.

## 📁 Full Data & Results
- Bias Test Results: `huggingface_evaluation_results.csv`
- Performance Test Results: `huggingface_performance_results.csv`

### **🏁 Conclusion**
The AI model demonstrates **good coherence, strong factual accuracy, and reasonable latency**. Improvements should focus on **reasoning skills and bias reduction**.

