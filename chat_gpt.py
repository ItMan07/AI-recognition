import openai

# from googletrans import Translator

# translator = Translator()

API_KEY = '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'
MODEL = 'text-davinci-003'
# MODEL = 'text-curie-001'
# MODEL = 'gpt-3.5-turbo-0301'
openai.api_key = API_KEY
work = 1
while work:
    prompt = input('Введите ваш запрос > ')
    # prompt = translator.translate(prompt, dest='ru', src='en')
    completion = openai.Completion.create(
        engine=MODEL,
        prompt=prompt,
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=0.5,
    )
    response = completion.choices[0].text
    # result = translator.translate(response, dest='ru', src='en')

    wrapped_text = []
    current_line = ""
    last_newline_pos = 0
    for i, char in enumerate(response):
        if char == '\n':
            wrapped_text.append(current_line)
            current_line = ""
            last_newline_pos = i
        elif i - last_newline_pos > 100 and char == " ":
            wrapped_text.append(current_line)
            current_line = ""
            last_newline_pos = i
        else:
            current_line += char

    if current_line:
        wrapped_text.append(current_line)

    for line in wrapped_text:
        print(line)
    print()
