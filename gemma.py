from openai import OpenAI
def generate(inter_text):
    client= OpenAI(
    base_url='http://localhost:8040/v1',
    api_key='abc',)
    response = client.chat.completions.create(
    model='google/gemma-3-4b-it',
    messages=[
        {'role': 'system', 'content': "оцени текст интервью"},
        {'role': 'user', 'content': inter_text}
    ],
    temperature = 0,
    max_tokens=1000,
    )
    return response.choices[0].message.content
