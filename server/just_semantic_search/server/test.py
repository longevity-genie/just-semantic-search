from litellm import embedding
import os
os.environ['OPENAI_API_KEY'] = ""
response = embedding(model='text-embedding-ada-002', input=["good morning from litellm"])