import streamlit as st
import requests
import json

# Configuration de la page
st.set_page_config(
    page_title="RAG IPCC AR6",
    page_icon="🌍",
    layout="wide"
)

# Titre
st.title("🌍 RAG Demo — IPCC AR6")
st.markdown("*Posez des questions sur les rapports IPCC AR6 (Ollama + LangChain)*")

# URL de l'API
API_URL = "http://localhost:8000"


# Zone de saisie
st.markdown("---")
question = st.text_input(
    "Votre question sur les rapports IPCC :",
    placeholder="Ex: Quels sont les principaux facteurs du changement climatique ?"
)

# Bouton de soumission
if st.button("🔍 Rechercher", type="primary") and question:
    with st.spinner("Recherche en cours..."):
        try:
            # Appel à l'API
            response = requests.post(
                f"{API_URL}/ask",
                json={"question": question},
                timeout=60
            )
            
            if response.ok:
                data = response.json()
                
                # Afficher la réponse
                st.markdown("### 💡 Réponse")
                st.markdown(data["answer"])
                
                # Afficher les sources
                if data.get("sources"):
                    st.markdown("---")
                    st.markdown("### 📚 Sources")
                    
                    for i, source in enumerate(data["sources"], 1):
                        with st.expander(f"Source {i}"):
                            st.markdown("**Extrait :**")
                            st.text(source.get("content", "N/A"))
                            
                            st.markdown("**Métadonnées :**")
                            st.json(source.get("metadata", {}))
            else:
                st.error(f"Erreur API : {response.status_code}")
                st.text(response.text)
                
        except requests.exceptions.Timeout:
            st.error("⏱️ Timeout : La requête a pris trop de temps")
        except Exception as e:
            st.error(f"❌ Erreur : {str(e)}")

# Sidebar avec informations
st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ À propos")
st.sidebar.markdown("""
Cette application utilise :
- **Ollama** (LLM local)
- **LangChain** (pipeline RAG)
- **ChromaDB** (base vectorielle)
- **FastAPI** (backend)
- **Streamlit** (interface)

**Documents :**
- IPCC AR6 WGI SPM
- IPCC AR6 SYR Full Volume
- IPCC AR6 SYR SPM
""")

# Exemples de questions
st.sidebar.markdown("---")
st.sidebar.markdown("### 💭 Questions exemples")
examples = [
    "Quels sont les principaux facteurs du changement climatique ?",
    "Que dit le rapport sur l'élévation du niveau de la mer ?",
    "Quelles sont les projections de température pour 2100 ?",
]
for ex in examples:
    if st.sidebar.button(ex, key=ex):
        st.rerun()