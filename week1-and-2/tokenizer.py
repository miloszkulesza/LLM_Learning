import tiktoken
import ollama

encoding = tiktoken.encoding_for_model("gpt-4.1-mini")
tokens = encoding.encode("Hi my name is Ed and I like banoffee pie")
print(f"Tokens: {tokens}")

for token_id in tokens:
    token_text = encoding.decode([token_id])
    print(f"{token_id} = {token_text}")

messages = [
    {"role": "system", "content": "You are a helpful assistant"},
    {"role": "user", "content": "Hi! I'm Ed!"}
    ]
print(messages)
response = ollama.chat(model="gpt-oss", messages=messages)
print(response['message']['content'])

messages = [
    {"role": "system", "content": "You are a helpful assistant"},
    {"role": "user", "content": "What's my name?"}
    ]
print(messages)
response = ollama.chat(model="gpt-oss", messages=messages)
print(response['message']['content'])

messages = [
    {"role": "system", "content": "You are a helpful assistant"},
    {"role": "user", "content": "Hi! I'm Ed!"},
    {"role": "assistant", "content": "Hi Ed! How can I assist you today?"},
    {"role": "user", "content": "What's my name?"}
    ]
print(messages)
response = ollama.chat(model="gpt-oss", messages=messages)
print(response['message']['content'])