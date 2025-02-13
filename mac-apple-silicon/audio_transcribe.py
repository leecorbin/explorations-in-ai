import os
import requests
from IPython.display import Markdown, display, update_display
from openai import OpenAI
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer, BitsAndBytesConfig
import torch
from dotenv import load_dotenv

DEVICE = "mps"

load_dotenv(override=True)
api_key = os.getenv('OPENAI_API_KEY')

if api_key and api_key.startswith('sk-proj-') and len(api_key)>10:
    print("API key looks good so far")
else:
    print("There might be a problem with your API key? Please visit the troubleshooting notebook!")
    
#MODEL = 'gpt-4o-mini'
openai = OpenAI()

AUDIO_MODEL = "whisper-1"
LLAMA = "meta-llama/Meta-Llama-3.1-8B-Instruct"

audio_filename = "audio.wav"

hf_token = os.getenv("HF_API_KEY")
login(hf_token, add_to_git_credential=True)

audio_file = open(audio_filename, "rb")
transcription = openai.audio.transcriptions.create(model=AUDIO_MODEL, file=audio_file, response_format="text")
print(transcription)

system_message = "You are given a sound file which could contain anything; audio from a YouTube video, a podcast or a lecture. Your task is to provide a summary of the audio content."
user_prompt = "Below is audio from a video, please provide a summary. Use Markdown to make it look attractive: " + transcription

messages = [
    {"role": "system", "content": system_message},
    {"role": "user", "content": user_prompt}
]

# quantization not working on my Mac
# quant_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_use_double_quant=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.bfloat16
# )

tokenizer = AutoTokenizer.from_pretrained(LLAMA)
tokenizer.pad_token = tokenizer.eos_token
inuts = tokenizer.apply_chat_template(messages, return_tensors="pt").to("mps")
streamer = TextStreamer(tokenizer)
model = AutoModelForCausalLM.from_pretrained(LLAMA, device_map="auto") #, quantization_config=quant_config)
outputs = model.generate(inputs=inuts, streamer=streamer, max_new_tokens=2000)

response = tokenizer.decode(outputs[0])

#display(Markdown(response))

with open(f"transcription.md", "w") as file:
        file.write(response)