from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
import shutil
import os

documents = [
    """Section 3.4 Configuring RIDC Port
    The RIDC port is used for socket-based communication with the content server.
    The default port value is 4444. To change it, navigate to Administration,
    then General Configuration, and update the SocketHostAddressSecurityFilter.""",
    
    """Section 3.5 Security Groups
    Security groups in WebCenter Content control access to documents.
    Each document belongs to one security group defined by dSecurityGroup metadata.
    Default security groups are Public, Secure, and Internal.""",
    
    """Section 4.1 IdcService Configuration
    IdcService is the core request dispatcher in WebCenter Content.
    All requests to the content server are routed through IdcService.
    Common services include GET_FILE, CHECKIN_NEW, and DOC_INFO.""",
]

#chunker 
splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=100,
    separators=["\n\n", "\n", ".", " "] )

chunks = splitter.create_documents(documents)
print(f"Created {len(chunks)} chunks")
for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1}: {chunk.page_content[:80]}...")


embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-V2")

if os.path.exists("./chroma_db"):
    shutil.rmtree("./chroma_db")

#vectore Store
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding = embeddings,
    persist_directory="./chroma_db"
)

print(f"vectore store created with {vectorstore._collection.count()} chunks")

query = "What is the default RIDC port value?"

results = vectorstore.similarity_search(query, k=2)

print(f"\nQuery: {query}")
print(f"\nTop {len(results)} results:")
for i, doc in enumerate(results):
    print(f"\nResult {i+1}:")
    print(doc.page_content)

# Initialize Ollama LLM
llm = OllamaLLM(model="mistral")

# Build context from retrieved chunks
context = "\n\n".join([doc.page_content for doc in results])

# Create prompt
prompt = f"""Answer the question based only on the context below. 
If the answer isn't in the context, say "I don't have that information."

Context:
{context}

Question: {query}

Answer:"""

# Get LLM response
answer = llm.invoke(prompt)

print(f"\n{'='*50}")
print(f"FINAL ANSWER:")
print(f"{'='*50}")
print(answer)


from datasets import Dataset
from ragas import evaluate
from ragas.metrics.collections import faithfulness, answer_relevancy, context_precision

from langchain_ollama import OllamaLLM
from langchain_ollama import OllamaEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

ragas_llm = LangchainLLMWrapper(OllamaLLM(model="mistral"))
ragas_embeddings = LangchainEmbeddingsWrapper(OllamaEmbeddings(model="nomic-embed-text"))

eval_data = {

    "question":[query],
    "answer":[answer],
    "contexts":[[doc.page_content for doc in results]],
    "ground_truth":["the default RIDC port value is 4444."]

}
dataset = Dataset.from_dict(eval_data)
print("\n Running  RAGS evaluation")
score = evaluate(dataset,metrics=[faithfulness,answer_relevancy,context_precision],llm=ragas_llm,embeddings=ragas_embeddings)
print(score)
