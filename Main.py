import ollama


# Обычный запрос
response = ollama.chat(
    model='gpt-oss:20b',
    messages=[
        {
            "role": "user",
            "content": "Hi!"
        }
    ]
)
print(response['message']['content'])

# stream = ollama.chat(
#     model='llama3.1',
#     messages=[{'role': 'user', 'content': prompt}],
#     stream=True
# )
# for chunk in stream:
#     print(chunk['message']['content'], end='', flush=True)