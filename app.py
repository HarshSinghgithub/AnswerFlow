import streamlit as st
from sentence_transformers import SentenceTransformer
import numpy as np
from transformers import pipeline
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
import time

API_KEY = st.secrets["API_KEY"]
pc = Pinecone(api_key=API_KEY)
index_name = 'semantic-search-fast'

embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
llm_model = "deepset/roberta-base-squad2"
QA = pipeline('question-answering', model=llm_model, tokenizer=llm_model)
db = pc.Index(index_name)

def get_answer(user_query):
  query_embedding = embedding_model.encode([user_query]).astype('float32')[0]

  answers = db.query(vector=query_embedding.tolist(), top_k=3, include_metadata=True)

  context = []
  for obj in answers['matches']:
    context.append(obj['metadata']['answer'])

  context = " ".join(context)

  QA_input = {
      'question': user_query,
      'context': context
  }
  res = QA(QA_input)

  return res['answer']



ques = st.text_input(label="Enter Question", value="")

if st.button("Answer"):
    if ques:
        st.spinner("Generating Answer")
        ans = get_answer(ques)
        if ans == " ":
           st.write("I Don't know the answer")
        else:
            st.write(ans)
    else:
        st.write("Enter Question")