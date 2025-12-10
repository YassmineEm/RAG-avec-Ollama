# app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter
import os
import sys
from typing import List, Dict, Any

app = FastAPI(
    title="RAG IPCC Climate API",
    description="Ask questions about IPCC climate reports using RAG",
    version="1.0.0"
)

# Vérifie si la base de données vectorielle existe
if not os.path.exists("vectordb"):
    print("ERREUR: Base de données vectorielle non trouvée!")
    print("Veuillez exécuter embeddings.py d'abord.")
    sys.exit(1)

# Initialisation du système
print("Chargement de la base de données vectorielle...")
try:
    embedding_fn = OllamaEmbeddings(
        model="nomic-embed-text:latest",
        base_url="http://localhost:11434"
    )
    
    # Charge la base de données vectorielle
    vectordb = Chroma(
        persist_directory="vectordb",
        embedding_function=embedding_fn
    )
    print("✓ Base de données vectorielle chargée")
    
    # Test de la base de données
    test_results = vectordb.similarity_search("climate", k=1)
    if test_results:
        print(f"✓ Base de données contient des documents")
    else:
        print("⚠️ Base de données vide ou problème de chargement")
        
except Exception as e:
    print(f"✗ Échec du chargement de la base de données: {e}")
    sys.exit(1)

# Crée le retriever
retriever = vectordb.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)

# Initialise le LLM
print("Initialisation du modèle de langage...")
try:
    llm = ChatOllama(
        model="llama3.2:3b",  # Vous pouvez changer pour "llama3.1:8b" ou "mistral"
        temperature=0.1,
        base_url="http://localhost:11434"
    )
    print("✓ Modèle de langage initialisé")
except Exception as e:
    print(f"✗ Échec de l'initialisation du modèle: {e}")
    print("Assurez-vous qu'Ollama est en cours d'exécution.")
    sys.exit(1)

# Crée le template de prompt
template = """Vous êtes un expert des rapports du GIEC (Groupe d'experts intergouvernemental sur l'évolution du climat).
Utilisez le contexte suivant des rapports du GIEC pour répondre à la question de manière précise et concise.

Si le contexte ne contient pas suffisamment d'informations pour répondre à la question, dites:
"Je ne peux pas répondre à cette question basée sur les rapports du GIEC disponibles."

Contexte:
{context}

Question: {question}

Fournissez une réponse claire et précise basée uniquement sur le contexte du GIEC ci-dessus:"""

prompt = PromptTemplate(
    template=template,
    input_variables=["context", "question"]
)

# Crée la chaîne RAG (sans utiliser RetrievalQA)
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Construit la chaîne RAG
rag_chain = (
    {
        "context": itemgetter("question") | retriever | format_docs,
        "question": itemgetter("question")
    }
    | prompt
    | llm
    | StrOutputParser()
)

print("\n" + "="*50)
print("✓ Système RAG prêt!")
print("="*50)
print("\nPoints de terminaison disponibles:")
print("  GET  /              - Informations API")
print("  POST /ask           - Poser une question")
print("  GET  /docs          - Documentation Swagger")
print("  GET  /health        - Vérification de santé")

# Modèles de requête/réponse
class QueryIn(BaseModel):
    question: str

class SourceDocument(BaseModel):
    content: str
    metadata: Dict[str, Any]

class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceDocument]

# Point de terminaison API
@app.post("/ask", response_model=QueryResponse)
async def ask(q: QueryIn):
    """Posez une question sur les rapports climatiques du GIEC."""
    print(f"\n[Requête] {q.question}")
    
    try:
        # Récupère les documents pertinents
        docs = retriever.invoke(q.question)
        
        # Exécute la chaîne RAG
        answer = rag_chain.invoke({"question": q.question})
        
        # Formate les sources
        sources = []
        for doc in docs:
            sources.append({
                "content": doc.page_content[:300] + ("..." if len(doc.page_content) > 300 else ""),
                "metadata": doc.metadata
            })
        
        print(f"[Rép] Longueur: {len(answer)} caractères")
        print(f"[Sources] Trouvé {len(sources)} documents pertinents")
        
        return {
            "answer": answer,
            "sources": sources
        }
        
    except Exception as e:
        print(f"[Erreur] {e}")
        raise HTTPException(status_code=500, detail=f"Erreur de traitement: {str(e)}")

@app.get("/", summary="Informations API")
async def root():
    """Obtenez des informations sur l'API."""
    return {
        "message": "API RAG GIEC Climat",
        "description": "Posez des questions sur les rapports du GIEC avec RAG",
        "endpoints": {
            "/ask": {
                "method": "POST",
                "description": "Posez des questions sur les rapports du GIEC"
            },
            "/docs": {
                "method": "GET", 
                "description": "Documentation interactive (Swagger UI)"
            },
            "/health": {
                "method": "GET",
                "description": "Vérification de santé du système"
            }
        },
        "status": "prêt"
    }

@app.get("/health", summary="Vérification de santé")
async def health_check():
    """Vérifiez si l'API fonctionne et est connectée à Ollama."""
    try:
        # Test de connexion Ollama
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        ollama_ok = response.status_code == 200
        
        # Test de la base de données vectorielle
        test_results = vectordb.similarity_search("climate", k=1)
        vectordb_ok = len(test_results) > 0
        
        # Test du modèle
        test_prompt = "Test"
        test_response = llm.invoke(test_prompt)
        llm_ok = len(test_response.content) > 0
        
        status = "healthy" if all([ollama_ok, vectordb_ok, llm_ok]) else "unhealthy"
        
        return {
            "status": status,
            "components": {
                "ollama": "connected" if ollama_ok else "disconnected",
                "vectordb": "loaded" if vectordb_ok else "empty",
                "llm": "ready" if llm_ok else "failed"
            },
            "timestamp": os.path.getmtime("vectordb") if os.path.exists("vectordb") else None
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "components": {
                "ollama": "unknown",
                "vectordb": "unknown", 
                "llm": "unknown"
            }
        }

if __name__ == "__main__":
    import uvicorn
    
    print("\nDémarrage du serveur FastAPI...")
    print("Serveur disponible à: http://localhost:8000")
    print("Documentation: http://localhost:8000/docs")
    print("Appuyez sur Ctrl+C pour arrêter\n")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=False
    )