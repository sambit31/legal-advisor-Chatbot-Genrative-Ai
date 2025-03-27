from src.helper import load_pdf, text_split, download_hugging_face_embeddings
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.environ.get('api_key')
url_Qdrant = os.environ.get('url_Qdrant')

# print(PINECONE_API_KEY)
# print(PINECONE_API_ENV)

extracted_data = load_pdf("Data/")
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()


#Initializing the Pinecone
qdrant_client = QdrantClient(
    url=url_Qdrant,
    api_key=api_key
)

# Define the collection name
collection_name = "legalbot-1"

# Create a collection (if it doesn’t exist)
if not qdrant_client.collection_exists(collection_name):
    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE)
    )
    print(f"Collection '{collection_name}' created successfully!")
else:
    print(f"Collection '{collection_name}' already exists.")


#Creating Embeddings for Each of The Text Chunks & storing
texts = [chunk.page_content for chunk in text_chunks]
metadatas = [chunk.metadata for chunk in text_chunks]

# Create Qdrant vector store
vector_store = Qdrant.from_texts(
    texts=texts,
    embedding=embeddings,
    metadatas=metadatas,
    url=url_Qdrant,
    api_key=api_key,
    collection_name=collection_name,
    force_recreate=True  # Ensures correct vector dimensions
)

print("Embeddings stored successfully!")
