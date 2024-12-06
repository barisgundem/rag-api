from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient , models 
from qdrant_client.http.models import Distance, VectorParams
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
import logging
from uuid import uuid4
import os
import tempfile



logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag-api")

# Constants
OLLAMA_URL = "http://ollama:11434"
MODEL = "bge-m3"
QDRANT_URL = "http://qdrant:6333"
COLLECTION_NAME = "rag_collection"

# Initialize Qdrant client and embedding model
client = QdrantClient(url=QDRANT_URL)
embeddings = OllamaEmbeddings(model=MODEL, base_url=OLLAMA_URL)

# Create collection if it doesn't exist
try:
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
    )
except Exception as e:
    print(f"Collection already exists or error in creating collection: {e}")

vector_store = QdrantVectorStore(
    client=client,
    collection_name=COLLECTION_NAME,
    embedding=embeddings,
)

# Initialize FastAPI app
app = FastAPI()

@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    file_id: str = Form(...),
    assistant_id: str = Form(None),
    conversation_id: str = Form(None),
):
    try:
        # Save the uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_path = temp_file.name
            temp_file.write(await file.read())

        # Load the PDF content using PyPDFLoader
        loader = PyPDFLoader(temp_path)
        documents = loader.load()

        if not documents:
            raise ValueError(f"No content extracted from the PDF with file_id: {file_id}.")

        # Add metadata to each document
        for doc in documents:
            doc.metadata.update({
                "file_id": file_id,
                "assistant_id": assistant_id,
                "conversation_id": conversation_id,
            })

        # Generate unique IDs for the documents and add them to the vector store
        uuids = [str(uuid4()) for _ in documents]
        vector_store.add_documents(documents=documents, ids=uuids)

        

        # Remove the temporary file
        os.remove(temp_path)

        return JSONResponse(
            status_code=200,
            content={"message": "PDF uploaded successfully", "document_ids": uuids},
        )

    except Exception as e:
        import traceback
        traceback.print_exc()  # Log stack trace for debugging
        return HTTPException(status_code=500, detail=f"Error uploading PDF: {str(e)}")




@app.post("/query")
async def query_vector_store(
    file_ids: str = Form(...),  # Comma-separated file IDs
    query: str = Form(...),
    k: int = Form(4),
):
    try:
        logger.info("Received query request.")
        logger.info(f"File IDs: {file_ids}")
        logger.info(f"Query: {query}")
        logger.info(f"Top-K: {k}")

        # Parse file IDs
        file_id_list = [fid.strip() for fid in file_ids.split(",")]
        logger.info(f"Parsed file IDs: {file_id_list}")

        # Build a filter for metadata using 'should' for multiple file IDs
        file_filter = models.Filter(
            should=[
                models.FieldCondition(
                    key="metadata.file_id",
                    match=models.MatchValue(value=fid),
                )
                for fid in file_id_list
            ]
        )
        logger.info(f"Generated metadata filter: {file_filter}")


        

        # Perform similarity search
        results = vector_store.similarity_search(query=query, k=k, filter=file_filter)
        logger.info(f"Similarity search results: {[res.page_content[:100] for res in results]}")

        similar_contents = [res.page_content for res in results]

        return JSONResponse(
            status_code=200,
            content={"similar_contents": similar_contents},
        )

    except Exception as e:
        logger.error(f"Error querying vector store: {str(e)}", exc_info=True)
        return HTTPException(status_code=500, detail=f"Error querying vector store: {str(e)}")

@app.post("/deleteFile")
async def delete_file(file_id: str = Form(...)):
    """
    Deletes all vectors associated with a specific file_id from the vector store.
    """
    try:
        logger.info(f"Received request to delete file with file_id: {file_id}")

        # Build a filter to select points by file_id
        file_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="metadata.file_id",
                    match=models.MatchValue(value=file_id),
                )
            ]
        )

        # Directly delete vectors using Qdrant client with FilterSelector
        client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=models.FilterSelector(filter=file_filter),
        )
        logger.info(f"Successfully deleted vectors for file_id: {file_id}")

        return JSONResponse(
            status_code=200,
            content={"message": f"Successfully deleted file_id: {file_id}"},
        )

    except Exception as e:
        logger.error(f"Error deleting file for file_id: {file_id}: {str(e)}", exc_info=True)
        return HTTPException(status_code=500, detail=f"Error deleting file: {str(e)}")



# Run the application
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
