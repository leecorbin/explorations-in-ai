#!/usr/bin/env python
# coding: utf-8

# ## Expert Knowledge Worker
# 
# ### A question answering agent that is an expert knowledge worker
# ### To be used by employees of Insurellm, an Insurance Tech company
# ### The agent needs to be accurate and the solution should be low cost.
# 
# This project will use RAG (Retrieval Augmented Generation) to ensure our question/answering assistant has high accuracy.
# 
# This first implementation will use a simple, brute-force type of RAG..
# 
# ### Sidenote: Business applications of this week's projects
# 
# RAG is perhaps the most immediately applicable technique of anything that we cover in the course! In fact, there are commercial products that do precisely what we build this week: nuanced querying across large databases of information, such as company contracts or product specs. RAG gives you a quick-to-market, low cost mechanism for adapting an LLM to your business area.

# In[1]:


# imports

import os
import glob
from dotenv import load_dotenv
import gradio as gr
from openai import OpenAI


# In[ ]:


# price is a factor for our company, so we're going to use a low cost model

MODEL = "gpt-4o-mini"


# In[ ]:


# Load environment variables in a file called .env

load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY', 'your-key-if-not-using-env')
openai = OpenAI()


# In[ ]:


# With massive thanks to student Dr John S. for fixing a bug in the below for Windows users!

context = {}

employees = glob.glob("knowledge-base/employees/*")

for employee in employees:
    name = employee.split(' ')[-1][:-3]
    doc = ""
    with open(employee, "r", encoding="utf-8") as f:
        doc = f.read()
    context[name]=doc


# In[ ]:


context["Lancaster"]


# In[ ]:


products = glob.glob("knowledge-base/products/*")

for product in products:
    name = product.split(os.sep)[-1][:-3]
    doc = ""
    with open(product, "r", encoding="utf-8") as f:
        doc = f.read()
    context[name]=doc


# In[ ]:


context.keys()


# In[ ]:


system_message = "You are an expert in answering accurate questions about Insurellm, the Insurance Tech company. Give brief, accurate answers. If you don't know the answer, say so. Do not make anything up if you haven't been provided with relevant context."


# In[ ]:


def get_relevant_context(message):
    relevant_context = []
    for context_title, context_details in context.items():
        if context_title.lower() in message.lower():
            relevant_context.append(context_details)
    return relevant_context          


# In[ ]:


get_relevant_context("Who is lancaster?")


# In[ ]:


get_relevant_context("Who is Avery and what is carllm?")


# In[ ]:


def add_context(message):
    relevant_context = get_relevant_context(message)
    if relevant_context:
        message += "\n\nThe following additional context might be relevant in answering this question:\n\n"
        for relevant in relevant_context:
            message += relevant + "\n\n"
    return message


# In[ ]:


print(add_context("Who is Alex Lancaster?"))


# In[ ]:


def chat(message, history):
    messages = [{"role": "system", "content": system_message}] + history
    message = add_context(message)
    messages.append({"role": "user", "content": message})

    stream = openai.chat.completions.create(model=MODEL, messages=messages, stream=True)

    response = ""
    for chunk in stream:
        response += chunk.choices[0].delta.content or ''
        yield response


# ## Now we will bring this up in Gradio using the Chat interface -
# 
# A quick and easy way to prototype a chat with an LLM

# In[ ]:


view = gr.ChatInterface(chat, type="messages").launch()


# In[ ]:




