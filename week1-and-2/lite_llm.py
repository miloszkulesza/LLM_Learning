from litellm import completion

tell_a_joke = [
    {"role": "user", "content": "Tell a joke for a student on the journey to becoming an expert in LLM Engineering"},
]
response = completion(model="ollama_chat/gpt-oss",
                      messages=tell_a_joke,
                      api_base="http://localhost:11434",
                      stream=True)
for chunk in response:
    content = chunk.choices[0].delta.content
    if content:
        print(content, end="", flush=True)

print("END")