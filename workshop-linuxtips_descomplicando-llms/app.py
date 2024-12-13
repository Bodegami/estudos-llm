import os
import gradio as gr
from pinecone import Pinecone, ServerlessSpec
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_openai.chat_models import ChatOpenAI
from langchain.vectorstores import Pinecone as pc
from langchain_openai import OpenAIEmbeddings

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(), override=True)

pinecone_api_key = os.environ.get('PINECONE_API_KEY')
pinecone_env = os.environ.get('PINECONE_ENV')
openai_apikey = os.environ.get('OPENAI_API_KEY')

embeddings = OpenAIEmbeddings()

pinecone = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
index_name = 'linuxtips'
index = pinecone.Index(index_name)

llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    api_key=openai_apikey
)

template = """
    Assistente é uma IA jurídica que tira dúvidas.
    Assistente elabora respostas simplificadas, com base no contexto fornecido.
    Assistente fornece referências extraídas do contexto abaixo. Não gere links ou referências adicionais.
    Ao final da resposta exiba no formato de lista as referências extraídas.
    Caso não consiga encontrar no contexto abaixo ou caso a pergunta não esteja relacionada do contexto jurídico,
    digas apenas 'Eu não sei!'.

    Pergunta: {query}

    Contexto: {context}
"""

prompt = PromptTemplate.from_template(template=template)

def search(query):
    docsearch = pc.from_existing_index(embedding=embeddings, index_name=index_name)
    docs = docsearch.similarity_search(query, k=3)
    context = docs[0].page_content + docs[1].page_content + docs[2].page_content
    resp = LLMChain(prompt=prompt, llm=llm)
    result = resp.invoke({'query': query, 'context': context})
    return result['text']


with gr.Blocks(title="IA jurídica", theme=gr.themes.Soft()) as ui:
    gr.Markdown("# Sou uma IA que tem a CLT como base de conhecimento")
    query = gr.Textbox(label='Faça a sua pergunta:', placeholder='EX: como funcionam as férias do trabalhador?')
    text_output = gr.Textbox(label="Resposta")
    btn = gr.Button("Perguntar")
    btn.click(fn=search, inputs=query, outputs=[text_output])
ui.launch(debug=True, share=False)    


