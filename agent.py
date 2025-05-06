import os
from dotenv import load_dotenv
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import (
    create_history_aware_retriever,
    create_retrieval_chain,
)
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langsmith import Client
from rich.console import Console
from rich.markdown import Markdown
from rich.prompt import Prompt
from langchain_community.chat_message_histories import ChatMessageHistory

import warnings
warnings.filterwarnings("ignore")
load_dotenv()

class PromptChatAgent:
    def __init__(self, system_instruction, collection_name="prompt_agente_knowledge", qdrant_path="./qdrant_db"):
        self.system_instruction = system_instruction
        self.embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        os.makedirs(qdrant_path, exist_ok=True)
        self.qdrant_client = QdrantClient(host="localhost", port=6338)
        self.collection_name = collection_name
        self._setup_vector_store()

        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
        self.chat_history = ChatMessageHistory()
        self._build_chain()

    def _setup_vector_store(self):
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

    def _build_chain(self):
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})

        # Prompt para reescrever a pergunta com base no histórico
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "Given a chat history and the latest user question "
             "which might reference context in the chat history, "
             "formulate a standalone question which can be understood "
             "without the chat history. Do NOT answer the question, just "
             "reformulate it if needed and otherwise return it as is."),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        history_aware_retriever = create_history_aware_retriever(
            self.llm, retriever, contextualize_q_prompt
        )

        # Prompt com {context}, necessário para o chain
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system",
             f"{self.system_instruction}\n\n"
             "Use o seguinte contexto para responder à pergunta do usuário. "
             "Se você não souber, diga que não sabe.\n\n{context}"),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

        question_answer_chain = create_stuff_documents_chain(self.llm, qa_prompt)

        self.rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    def add_knowledge_from_file(self, file_path):
        loader = TextLoader(file_path, encoding="utf-8")
        docs = loader.load()
        text_splitter = SemanticChunker(OpenAIEmbeddings())
        texts = text_splitter.create_documents([docs[0].page_content])
        self.vectorstore.add_documents(texts)
        print(f"Conhecimento de {file_path} adicionado à base de dados persistente.")

    def chat(self, user_input):
        inputs = {
            "input": user_input,
            "chat_history": self.chat_history.messages,
        }
        response = self.rag_chain.invoke(inputs)
        answer = response["answer"]
        self.chat_history.add_user_message(user_input)
        self.chat_history.add_ai_message(answer)
        return answer

    def reset_knowledge_base(self):
        if self.qdrant_client.collection_exists(self.collection_name):
            self.qdrant_client.delete_collection(self.collection_name)
            self._setup_vector_store()
            print(f"Base de conhecimento '{self.collection_name}' resetada.")



def run_chat_interface():
    console = Console()
    client = Client()
    prompt = client.pull_prompt("gerador-prompt", include_model=True)
    system_instruction = prompt.first.messages[0].prompt.template
    agent = PromptChatAgent(system_instruction)

    console.print("[bold cyan]🤖 Agente de Prompts Iniciado![/bold cyan]")
    console.print("[green]Digite sua pergunta ou use comandos:[/green]")
    console.print("• /add caminho.txt → adiciona conhecimento")
    console.print("• /reset → reseta base de conhecimento")
    console.print("• /sair → encerra o chat\n")

    while True:
        user_input = Prompt.ask("[bold yellow]Você[/bold yellow]")

        if user_input.strip().lower() == "/sair":
            console.print("[red]Encerrando o agente...[/red]")
            break

        elif user_input.startswith("/add "):
            file_path = user_input[5:].strip()
            try:
                agent.add_knowledge_from_file(file_path)
                console.print(f"[green]✅ Conhecimento de '{file_path}' adicionado![/green]")
            except Exception as e:
                console.print(f"[red]Erro ao adicionar '{file_path}':[/red] {e}")
            continue

        elif user_input.strip().lower() == "/reset":
            agent.reset_knowledge_base()
            console.print("[yellow]⚠️ Base de conhecimento resetada.[/yellow]")
            continue

        try:
            response = agent.chat(user_input)
            console.print(Markdown(f"**Agente:** {response}"))
        except Exception as e:
            console.print(f"[red]Erro ao processar a mensagem:[/red] {e}")

if __name__ == "__main__":
    run_chat_interface()