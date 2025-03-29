import os
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup
from langchain_experimental.text_splitter import SemanticChunker
from langchain_cohere import CohereEmbeddings
from langchain.vectorstores import FAISS
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from anthropic import Anthropic

load_dotenv()


class DocumentLoader:
    def __init__(self, urls):
        self.urls = urls

    def scrape_webpages(self):
        docs = []
        for url in self.urls:
            response = requests.get(url)
            soup = BeautifulSoup(response.content, "html.parser")
            docs.append(soup.get_text())
        return "\n".join(docs)


class KnowledgeBase:
    def __init__(self, embedding_model, faiss_path, cohere_key):
        self.embedding_model = embedding_model
        self.faiss_path = faiss_path
        self.cohere_key = cohere_key

    def build_index(self, combined_docs):
        splitter = SemanticChunker(embeddings=self.embedding_model)
        chunks = splitter.split_text(combined_docs)
        print(f"Generated {len(chunks)} semantic chunks.")
        if os.path.exists(self.faiss_path):
            print("Loading existing FAISS index...")
            vectorstore = FAISS.load_local(self.faiss_path, self.embedding_model, allow_dangerous_deserialization=True)

        else:
            print("Creating new FAISS index...")
            vectorstore = FAISS.from_texts(chunks, self.embedding_model)
            vectorstore.save_local(self.faiss_path)
        compressor = CohereRerank(cohere_api_key=self.cohere_key, top_n=5)
        retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=vectorstore.as_retriever()
        )
        return retriever


class PromptFactory:
    @staticmethod
    def get_prompt():
        return PromptTemplate(
            input_variables=["context", "question"],
            template="""
            You are a senior WhatsApp Flow Architect specializing in building production-ready Flow JSONs.

            Use the following documentation to answer user queries and output WhatsApp Flow JSON ready to be deployed.

            CONTEXT:
            {context}

            USER REQUEST:
            {question}

            Guidelines:
            - Only return the JSON output.
            - Use placeholders like {{variable}} if the user input is missing.
            - Make the Flow dynamic where needed (e.g., dynamic buttons).

            JSON:
            """
        )


class FlowComposerAgent:
    URLS = [
        "https://developers.facebook.com/docs/whatsapp/flows/reference/flowjson",
        "https://developers.facebook.com/docs/whatsapp/flows/reference/components"
    ]
    
    def __init__(self):
        load_dotenv()
        embedding_model = CohereEmbeddings(
            model="embed-english-v3.0",
            cohere_api_key=os.getenv("cohere_api_key")
        )
        
        loader = DocumentLoader(self.URLS)
        docs = loader.scrape_webpages()
        
        kb = KnowledgeBase(
            embedding_model, 
            faiss_path="faiss_index", 
            cohere_key=os.getenv("cohere_api_key")
        )
        self.retriever = kb.build_index(docs)
        
        self.client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.model = "claude-3-sonnet-20240229"
        self.prompt_template = PromptFactory.get_prompt()

    def compose_flow(self, user_query):
        docs = self.retriever.get_relevant_documents(user_query)
        context = "\n\n".join([d.page_content for d in docs])
        formatted_prompt = self.prompt_template.format(context=context, question=user_query)
        
        response = self.client.messages.create(
            model=self.model,
            messages=[{"role": "user", "content": formatted_prompt}],
            system="You are a senior WhatsApp Flow Architect specializing in building production-ready Flow JSONs.",
            max_tokens=4096,
            temperature=0
        )
        return response.content[0].text.strip()


# Main orchestration
if __name__ == "__main__":
    agent = FlowComposerAgent()