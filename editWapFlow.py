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

class FlowTransformerAgent:
    URLS = [
        "https://developers.facebook.com/docs/whatsapp/flows/reference/flowjson",
        "https://developers.facebook.com/docs/whatsapp/flows/reference/components",
        "https://reactflow.dev/docs/api/nodes/node-options/"
    ]
    
    def __init__(self):
        self.embedding_model = CohereEmbeddings(
            model="embed-english-v3.0",
            cohere_api_key=os.getenv("cohere_api_key")
        )
        
        # Check if FAISS index exists
        if not os.path.exists("flow_converter_index"):
            loader = DocumentLoader(self.URLS)
            docs = loader.scrape_webpages()
            
            kb = KnowledgeBase(
                self.embedding_model, 
                faiss_path="flow_converter_index", 
                cohere_key=os.getenv("cohere_api_key")
            )
            self.retriever = kb.build_index(docs)
        else:
            vectorstore = FAISS.load_local("flow_converter_index", self.embedding_model, allow_dangerous_deserialization=True)
            compressor = CohereRerank(cohere_api_key=os.getenv("cohere_api_key"), top_n=5)
            self.retriever = ContextualCompressionRetriever(
                base_compressor=compressor,
                base_retriever=vectorstore.as_retriever()
            )
        
        self.client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.model = "claude-3-sonnet-20240229"
        self.prompt_template = self._get_conversion_prompt()

    def _get_conversion_prompt(self):
        return PromptTemplate(
            input_variables=["context", "react_flow_json"],
            template="""
            You are a Flow Conversion Specialist that transforms React Flow JSON into WhatsApp Flow JSON.

            Use the following documentation to properly convert the React Flow JSON:

            CONTEXT:
            {context}

            REACT FLOW JSON:
            {react_flow_json}

            Guidelines:
            - Analyze the React Flow structure and map it to equivalent WhatsApp Flow components
            - Preserve the flow logic and connections
            - Only return the WhatsApp Flow JSON
            - Ensure all required WhatsApp Flow properties are included
            - Map React Flow node types to appropriate WhatsApp components

            WHATSAPP FLOW JSON:
            """
        )

    def transform_flow(self, react_flow_json):
        # Get relevant documentation for the conversion
        docs = self.retriever.get_relevant_documents(str(react_flow_json))
        context = "\n\n".join([d.page_content for d in docs])
        
        # Format prompt with context and React Flow JSON
        formatted_prompt = self.prompt_template.format(
            context=context, 
            react_flow_json=react_flow_json
        )
        
        # Get conversion response from Claude
        response = self.client.messages.create(
            model=self.model,
            messages=[{"role": "user", "content": formatted_prompt}],
            system="You are a Flow Conversion Specialist that transforms React Flow JSON into WhatsApp Flow JSON.",
            max_tokens=4096,
            temperature=0
        )
        return response.content[0].text.strip()

# Main execution
if __name__ == "__main__":
    agent = FlowTransformerAgent()
    # Example usage:
    # react_flow = {...}  # Your React Flow JSON
    # whatsapp_flow = agent.transform_flow(react_flow)
