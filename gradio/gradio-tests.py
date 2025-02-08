import os
import requests
from bs4 import BeautifulSoup
from typing import List
from dotenv import load_dotenv
from openai import OpenAI
import google.generativeai
import anthropic
import gradio as gr # oh yeah!

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
claude = anthropic.Anthropic()
google.generativeai.configure()

system_message = "You are a helpful assistant that responds in markdown"

def message_gpt(prompt):
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt}
      ]
    completion = openai.chat.completions.create(
        model='gpt-4o-mini',
        messages=messages,
    )
    return completion.choices[0].message.content

def shout(text):
    print(f"Shout has been called with input {text}")
    return text.upper()

#gr.Interface(fn=shout, inputs="textbox", outputs="textbox").launch()
# gr.Interface(fn=shout, inputs="textbox", outputs="textbox", flagging_mode="never").launch(share=True)
#gr.Interface(fn=shout, inputs="textbox", outputs="textbox", flagging_mode="never").launch(inbrowser=True)

# force_dark_mode = """
# function refresh() {
#     const url = new URL(window.location);
#     if (url.searchParams.get('__theme') !== 'dark') {
#         url.searchParams.set('__theme', 'dark');
#         window.location.href = url.href;
#     }
# }
# """
# gr.Interface(fn=shout, inputs="textbox", outputs="textbox", flagging_mode="never", js=force_dark_mode).launch()

# view = gr.Interface(
#     fn=message_gpt,
#     inputs=[gr.Textbox(label="Your message:")],
#     outputs=[gr.Markdown(label="Response:")],
#     flagging_mode="never"
# )
# view.launch()

def stream_gpt(prompt):
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt}
      ]
    stream = openai.chat.completions.create(
        model='gpt-4o-mini',
        messages=messages,
        stream=True
    )
    result = ""
    for chunk in stream:
        result += chunk.choices[0].delta.content or ""
        yield result

# view = gr.Interface(
#     fn=stream_gpt,
#     inputs=[gr.Textbox(label="Your message:")],
#     outputs=[gr.Markdown(label="Response:")],
#     flagging_mode="never"
# )
# view.launch()

def stream_claude(prompt):
    result = claude.messages.stream(
        model="claude-3-haiku-20240307",
        max_tokens=1000,
        temperature=0.7,
        system=system_message,
        messages=[
            {"role": "user", "content": prompt},
        ],
    )
    response = ""
    with result as stream:
        for text in stream.text_stream:
            response += text or ""
            yield response

# view = gr.Interface(
#     fn=stream_claude,
#     inputs=[gr.Textbox(label="Your message:")],
#     outputs=[gr.Markdown(label="Response:")],
#     flagging_mode="never"
# )
# view.launch()


def stream_model(prompt, model):
    if model=="GPT":
        result = stream_gpt(prompt)
    elif model=="Claude":
        result = stream_claude(prompt)
    else:
        raise ValueError("Unknown model")
    yield from result

view = gr.Interface(
    fn=stream_model,
    inputs=[gr.Textbox(label="Your message:"), gr.Dropdown(["GPT", "Claude"], label="Select model", value="GPT")],
    outputs=[gr.Markdown(label="Response:")],
    flagging_mode="never"
)
view.launch()






