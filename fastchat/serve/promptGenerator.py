from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder

import numpy as np
import os
import pandas as pd
import re

embeddings=HuggingFaceEmbeddings(model_name='hiiamsid/sentence_similarity_spanish_es')

class PromptGenerator:
    def __init__(self, knowledge_dir = 'documents', k = 3, chunk_length = 600):
        self.knowledge_dir = knowledge_dir  # Path to the knowledge base
        self.k = k  # The number of most related paragraphs to be included in the prompt
        self.chunk_length = chunk_length  # Length of each chunk of text
        self._read_paragraphs()
        print('Loading CrossEncoder...')
        self.cross_encoder = CrossEncoder('cross-encoder/mmarco-mMiniLMv2-L12-H384-v1')

    # Prompt generation
    def _generate_prmopt(self, docs, question):
        input_text = ''
        for doc in docs:
            input_text += doc + '\n'
        #input_text += '\n' + question   
        return input_text

    def _read_paragraphs(self):
        print('Loading embeddings...')
        embeddings=HuggingFaceEmbeddings(model_name='hiiamsid/sentence_similarity_spanish_es')
        print('Loading documents...')
        loader =DirectoryLoader(self.knowledge_dir, show_progress=True, use_multithreading=True, glob='*.txt')
        data = loader.load()
        print('Splitting documents...')
        documents = []
        for d in data:
            documents += [
                re.sub(r'\s+', ' ', document).strip() \
                    for document in d.page_content.split('\n\n')
                ]
        print('Creating vector store...')
        self.docsearch = Chroma.from_texts(documents, embeddings)

    def get_prompt(self, question):
        docs=[doc.page_content for doc in self.docsearch.similarity_search(question, k=100)]
        scores = self.cross_encoder.predict([[question, doc] for doc in docs])
        scores = np.array(scores)[scores > 0.5]
        top_k_indices = scores.argsort()[::-1][:self.k]
        docs = [docs[i] for i in top_k_indices]
        if len(docs) == 0:
            prompt = ''
        else:
            prompt = self._generate_prmopt(docs, question) \
                + "\nResponde a la siguiente pregunta utilizando únicamente el contexto, " \
                + "pero no menciones que te basas en él. "
        return prompt


if __name__ == '__main__':
    promptGenerator = PromptGenerator()
    prompt = promptGenerator.get_prompt("¿Qué tengo que hacer para empadronarme en Calatayud?")
    print(prompt)
