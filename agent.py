import os
from dotenv import load_dotenv
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

load_dotenv()

class PromptAgent:
    def __init__(self, system_instruction, collection_name="langchain_knowledge", qdrant_path="./qdrant_db"):
        self.system_instruction = system_instruction
        self.embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # Criar pasta para o banco de dados se não existir
        os.makedirs(qdrant_path, exist_ok=True)
        
        # Inicializar QDrant com armazenamento persistente
        self.qdrant_client = QdrantClient(path=qdrant_path)
        self.collection_name = collection_name

        self._setup_vector_store()
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

    def _setup_vector_store(self):
        # Verificar se a coleção já existe antes de tentar criá-la
        if not self.qdrant_client.collection_exists(self.collection_name):
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE),
            )

        self.vectorstore = QdrantVectorStore(
            client=self.qdrant_client,
            collection_name=self.collection_name,
            embedding=self.embedding_model,
        )

    def add_knowledge_from_file(self, file_path):
        loader = TextLoader(file_path, encoding="utf-8")
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        texts = text_splitter.create_documents([docs[0].page_content])
        self.vectorstore.add_documents(texts)
        print(f"Conhecimento de {file_path} adicionado à base de dados persistente.")

    def generate_prompt(self, user_input):
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})
        rag_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=retriever,
            chain_type="stuff",
            return_source_documents=False,
            chain_type_kwargs={
                "prompt": self._build_custom_prompt()
            }
        )

        result = rag_chain.invoke({"query": user_input})
        return result

    def _build_custom_prompt(self):
        from langchain.prompts import PromptTemplate

        template = (
            f"{self.system_instruction}\n"
            "Com base no contexto abaixo, responda a pergunta do usuário.\n\n"
            "Contexto:\n{context}\n\n"
            "Pergunta: {question}\nResposta:"
        )

        return PromptTemplate(
            input_variables=["context", "question"],
            template=template
        )

    def reset_knowledge_base(self):
        """Método opcional para resetar a base de conhecimento"""
        if self.qdrant_client.collection_exists(self.collection_name):
            self.qdrant_client.delete_collection(self.collection_name)
            self._setup_vector_store()
            print(f"Base de conhecimento '{self.collection_name}' resetada.")

# Exemplo de uso
if __name__ == "__main__":
    agent = PromptAgent("Você é um especialista em gerar prompts otimizados para modelos de linguagem.")

    # Adicionar conhecimento (só será adicionado se ainda não estiver na base)
    agent.add_knowledge_from_file("teste.txt")

    entrada = "Crie um prompt para gerar um avatar 3D realista para um jogo de fantasia."
    resposta = agent.generate_prompt(entrada)
    print(resposta["result"])  # Acesso correto ao resultado