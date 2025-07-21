from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
# import regex as re
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
import requests 
from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
import streamlit as st
# from langchain_huggingface import HuggingFaceEndpoint
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
st.header("YOUTUBE BOT")
q=st.text_input("Enter Your Query ")
lang=st.selectbox("Language",options=["HINDI","TELUGU","FRENCH","GREEK"])
# os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
try:
    # app=
    docs=YouTubeTranscriptApi.get_transcript("u6qDSFxY4iw")
    docss=" ".join(doc['text'] for doc in docs)
except TranscriptsDisabled:
    print("Empty")
except NoTranscriptFound:
    print("Not Available")
split=RecursiveCharacterTextSplitter(chunk_size=500)
splitted=split.split_text(docss)
embeddin=HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-MiniLM-L6-v2",
)
llm=ChatGroq(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    api_key="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"
)
print(q)
cont=FAISS.from_texts(splitted,embedding=embeddin)
retr=cont.as_retriever()
da=retr.invoke(q)
# context="\n\n".join(da)
prompt=PromptTemplate(
    template="Answer the question:{query} from the following paragraph in a deep way:{context} and translate it to the following language:{language}",
    input_variables=['query','context','language']
)
llms=prompt|llm
formatted_prompt = prompt.format(query=q, context=cont,language=lang)
resp=llms.stream({"query":q,"context":da,"language":lang})
print(resp)
st.write_stream(r.content for r in resp)