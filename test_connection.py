from openai import OpenAI
import os


client = OpenAI(base_url="http://127.0.0.1:50124/v1", api_key="x")

try:
    response = client.chat.completions.create(
        model="qwen3-0.6b-generic-gpu:2", 
        messages=[{"role": "user", "content": "Hello"}]
    )
    print("Başarılı! Yanıt:", response.choices[0].message.content)
except Exception as e:
    print("Hata:", e)