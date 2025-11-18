import boto3 
import json 
import chromadb
from chromadb.utils import embedding_functions

bedrock = boto3.client('bedrock-runtime', region_name='us-east-1')
EMBEDDING_MODEL = "amazon.titan-embed-text-v2:0"
TEXT_GENERATION_MODEL = "anthropic.claude-3-sonnet-20240229-v1:0"

def get_bedrock_embedding(text):
    """
    Generate embedding for the given text using Bedrock embedding model.
    """
    # print(f"Generating embedding for text: {text}")
    request_body = {
        "inputText": text,
    }
    response = bedrock.invoke_model(
        modelId=EMBEDDING_MODEL,
        body=json.dumps(request_body),
    )
    response_body = json.loads(response['body'].read())
    return response_body['embedding']

def generate_text(prompt):
    """
    Generate text for the given prompt using Bedrock text generation model.
    """
    # print(f"Generating text for prompt: {prompt}")
    request_body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1024,
        "temperature": 0,
        'top_p': 0.002,
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}]
            }
        ]
    }
    response = bedrock.invoke_model(
        modelId=TEXT_GENERATION_MODEL,
        body=json.dumps(request_body),
    )
    response_body = json.loads(response['body'].read())
    generated_text = response_body['content'][0]['text']
    return generated_text

chroma_client = chromadb.Client()

def bedrock_embedding_fn(texts):
    vectors = []
    for t in texts:
        vec = get_bedrock_embedding(t)
        vectors.append(vec)
    return vectors

from chromadb import Documents, EmbeddingFunction, Embeddings

class MyEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        # embed the documents somehow
        embeddings = bedrock_embedding_fn(input)
        return embeddings
    
collection = chroma_client.create_collection(
    name="bedrock_docs",
    embedding_function=MyEmbeddingFunction()
)

def add_documents(docs):
    """
    Add documents to the ChromaDB collection after generating embeddings.
    """
    # print(f"Adding documents: {docs}")
    collection.add(
        documents=docs,
        ids=[f"doc_{i}" for i in range(len(docs))],
    )

sample_docs = [ "Amazon Bedrock is a fully managed foundation model service.", \
               "RAG systems combine retrieval and generation for improved responses.", \
               "Embeddings are vector representations of text in high-dimensional space.", \
               "Chroma is an efficient vector store for building AI applications.", \
               "Foundation models can be fine-tuned for specific tasks and domains." ] 

add_documents(sample_docs)

def rag_generate(query, top_k = 2):
    results = collection.query(
        query_texts=[query],
        n_results=top_k
    )
    context = "\n".join(results['documents'][0]) 
    prompt = f"""Human: Given the following context, please answer the question.
                Context: {context}
                Question: {query}
                Assistant: Based on the provided context, I can answer as follows:
            """
    response = generate_text(prompt)
    return response

def generate_without_rag(query):
    prompt = f"Human: {query}\n\nAssistant:" 
    return generate_text(prompt)

# Example usage
if __name__ == "__main__":
    # query = "How does Amazon Bedrock relate to RAG systems?"
    # response = rag_generate(query) 
    # print(f"Query: {query}") 
    # print(f"Response: {response}")
    test_queries = [ "What are embeddings used for in AI?", 
                    "Explain the benefits of using RAG in AI applications.", 
                    "How does Amazon Bedrock support foundation models?" ]
    for query in test_queries: 
        print(f"\nQuery: {query}") 
        print(f"\nRAG Response: {rag_generate(query)}") 
        print(f"\nNon-RAG Response: {generate_without_rag(query)}") 
        print("\n" + "="*50)