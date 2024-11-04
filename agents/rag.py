# RAG
import pandas as pd
import requests
import json
from typing import List, Dict, Any
import faiss
import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore

from langchain_core.documents import Document
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

import torch

from openai import OpenAI

url = "https://innoglobalhack-general.olymp.innopolis.university/v1/chat/completions"
model_name = "multilingual-e5-large-instruct/"


class RagClass:
    def __init__(self, table_path: str = 'df_rag_year.csv', vector_db_path: str = None) -> None:
        
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.model = SentenceTransformer(model_name).to(self.device)
        
        if vector_db_path is None:
            collection = self._init_faiss_db(
                                            model_name="multilingual-e5-large-instruct/",
                                            path='faiss-db/'
                                            )
            
            self.vector_db = self._load_raw_data(collection=collection, table=table_path)
        else:
            self.vector_db = self._load_faiss_db(path=vector_db_path)

    
    def rag(self, answer: str = "Hello, name three president in US", mode: str = 'textonly', n_results: int = 10, year = None) -> str:
        
        filter_dict = {'year': year} if year is not None else {}
        
        retrival_chunks = self.querry(user_question=answer, n_results=n_results, filter=filter_dict)
        
        text_for_llm = self.generate_prompt(answer, [chunk['content'] for chunk in retrival_chunks], mode=mode)
        answer = self._ask_llm(question=text_for_llm)
        
        return answer #+ "\n\n\n" + str(retrival_chunks)
        
        
    def rag_for_data_filling(self, metric: str = 'Прибыль', year: int = None) -> str:
        filter_dict = {'year': year} if year is not None else {}
        
        retrival_chunks = self.querry(user_question=f"{metric} за {str(year)}", n_results=10, filter=filter_dict)
        
        retrival_chunks = [( str(doc['content']) + '\n\n' + str(doc['table'])) for doc in retrival_chunks if doc['is_table']]
        
        text_for_llm = self.generate_prompt(f"Верни только значение метрики {metric} за {year} год, одним числом.", retrival_chunks, mode='textonly')
        
        answer = self._ask_llm(question=text_for_llm)
        
        return answer 
        
        
    def querry(self, 
            user_question: str = "Hello, name three president in US", 
            n_results: int = 10,
            filter: dict = {}
            
        ) -> List[dict]:
        
        retrival_chunks = self.vector_db.search(
            user_question,
            filter=filter,
            k = n_results,
            fetch_k = 100,
            search_type='similarity'
        )
        
        answer = [{'page': retrival_chunk.metadata['page'], 
                   'filename': retrival_chunk.metadata['file'],
                   'is_table': retrival_chunk.metadata['is_table'],
                   'table': retrival_chunk.metadata['table'],
                   'content': retrival_chunk.page_content} 
                  
                  for retrival_chunk in retrival_chunks]
        
        return answer
        
    def generate_prompt(self, user_question, chunks, plot: str = None, mode: str ='textonly') -> str:
    
        prompt_template = None
        text_chunk = '\n\n'.join(chunks)
        
        if mode == 'textonly':
            with open('agents/chunks_prompt.txt', 'r') as f:  
                prompt_template = f.read()
                prompt = prompt_template.replace('{CHUNK}', text_chunk).replace('{USER_QUESTION}',user_question)
        else:
            with open('agents/plot_prompt.txt', 'r') as f:
                prompt_template = f.read()
                prompt = prompt_template.replace('{CHUNK}', text_chunk).replace('{PLOT}', plot)
        
        return prompt

    
    def reload_vector_db(self, path: str = 'faiss-db/'):
        vector_store = FAISS.load_local(folder_path=path, embeddings=self._function_for_embeddings)
        print("Succecfully loaded vector data base")
        return vector_store

    def _function_for_embeddings(self, text):
        return np.array(self.model.encode(text, convert_to_tensor=True, normalize_embeddings=True).detach().cpu())
    
    def _function_for_embeddings_2(self, text):
        return np.random.rand(1024)
        
    def _init_faiss_db(
        self,
        model_name = "multilingual-e5-large-instruct/",
        path = 'faiss-db/',
        name = "mts-doc-db"
        ):
        embed_func = self._function_for_embeddings
        
        index = faiss.IndexFlatL2(1024)

        vector_store = FAISS(
            embedding_function=embed_func,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )
        vector_store.save_local(folder_path=path)
        return vector_store

    
    def _load_raw_data(self, 
                      collection, 
                      table: str = "tables/raw.csv"
        ):
    
        if isinstance(table, str):
            csv_table = pd.read_csv(table)
        else:
            csv_table = table
            
        print("Loading table...")
        for i, row in csv_table.iterrows():
            print(f"{i/len(csv_table)*100:.2f} %", end='\r')
            if row['is_table']: 
                continue
            
            chunked_data = self.create_chunks_langchain([row['extracted_text']])
            collection.add_texts( 
                chunked_data, 
                metadatas=[{ 
                    'file': row['pdf_path'], 
                    'index': i, 
                    'page': row['page_num'], 
                    'year': row['year'],
                    'is_table': row['is_table']
                
                } for _ in range(len(chunked_data))]
            )

        print(f"Successfully loaded {len(csv_table)} documents")
        return collection
    
    def _ask_llm(self, question: str = "Hello, name three president in US") -> str:
        img_extra = {
                "model": "meta-llama/Llama-3.2-11B-Vision-Instruct",
                "add_special_tokens": True,
                "messages": [
                    {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"{question}"
                        }
                        ]
                    }
                ]
        }
        
        resp = requests.post(url,
                            json=img_extra
                            )
        return resp.json()['choices'][0]['message']['content']
    
    def _ask_llm_gpt(self, question):
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Указываем модель
            messages=[
                {"role": "user", "content": question}
            ],
            max_tokens=100,  # Максимальное количество токенов в ответе
            temperature=0.7  # Температура для настройки креативности ответа
        )
        if isinstance(response, str):
            return json.loads(response)["choices"][0]["message"]["content"]
        else:
            response = response.to_dict()['choices'][0]['message']['content']
            return response
    
    def _load_faiss_db(self, path: str = 'faiss-db/'):
        vector_store = FAISS.load_local(folder_path=path, embeddings=self._function_for_embeddings_2, allow_dangerous_deserialization=True)
        return vector_store

    def reload_model(self, model_name: str = "multilingual-e5-large-instruct/") -> None:
        self.model = SentenceTransformer(model_name).to(self.device)
        
    def create_chunks(self, document: str, chunk_size=1000, overlap=200) -> List[str]:
        return [document[i : i + chunk_size] for i in range(0, len(document), chunk_size - overlap)]

    def create_chunks_langchain(self, documents: List[str], chunk_size=1000, overlap=200) -> List[str]:
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
        chunks = [doc.page_content for doc in splitter.split_documents([Document(page_content=document) for document in documents])]

        return chunks

    
def main():
    # rag = RagClass(vector_db_path='faiss-db')
    
    # user_question = ''
    # while user_question != "exit":
    #     user_question = input("Put your request: ")
    #     print(user_question)
        
    #     answer = rag.rag_for_data_filling(metric=user_question, year=2023)
        
    #     print(f"Answer: \n{answer}")
    pass

    
if __name__ == '__main__':
    main()