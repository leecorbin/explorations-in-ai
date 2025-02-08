import os
from dotenv import load_dotenv
from openai import OpenAI
import gradio as gr

load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')
anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
google_api_key = os.getenv('GOOGLE_API_KEY')

if openai_api_key:
    print(f"OpenAI API Key exists and begins {openai_api_key[:8]}")
else:
    print("OpenAI API Key not set")
    
if anthropic_api_key:
    print(f"Anthropic API Key exists and begins {anthropic_api_key[:7]}")
else:
    print("Anthropic API Key not set")

if google_api_key:
    print(f"Google API Key exists and begins {google_api_key[:8]}")
else:
    print("Google API Key not set")

openai = OpenAI()
MODEL = 'gpt-4o-mini'

system_message = "You are a helpful assistant in an electronics store. You should try to gently encourage \
the customer to try items that are on sale. Tablets are 20% off, and some other items are 10% off. \
Encourage the customer to buy a tablet if they are unsure what to get."

system_message += "\nIf the customer asks for computers, you should respond that computers are not on sale today, \
but remind the customer to look at tablets!"

def chat(message, history):

    relevant_system_message = system_message
    if 'phone' in message:
        relevant_system_message += " The store does not sell phones; if you are asked for phones, be sure to point out other items on sale."
    
    messages = [{"role": "system", "content": relevant_system_message}] + history + [{"role": "user", "content": message}]

    stream = openai.chat.completions.create(model=MODEL, messages=messages, stream=True)

    response = ""
    for chunk in stream:
        response += chunk.choices[0].delta.content or ''
        yield response

gr.ChatInterface(fn=chat, type="messages").launch()
