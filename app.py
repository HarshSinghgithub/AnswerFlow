import streamlit as st
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import json
from transformers import pipeline

embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
llm_model = "deepset/roberta-base-squad2"
QA = pipeline('question-answering', model=llm_model, tokenizer=llm_model)
db = faiss.read_index("faiss_index.index")

with open('answers.json', 'r') as f:
    answers = json.load(f)

        
def get_answer(user_query):
  query_embedding = embedding_model.encode([user_query]).astype('float32')
  k = 2
  D, I = db.search(query_embedding, k)

  retrieved_answers = [answers[i] for i in I[0]]

  context = " ".join(retrieved_answers)

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
        st.write(ans)
    else:
        st.write("Enter Question")