# RAG Ollama Lab

Ce projet met en place un système **RAG (Retrieval-Augmented Generation)** utilisant **Ollama** pour répondre aux questions sur les documents IPCC.

---

## 🛠️ Setup

1. **Installer Ollama et démarrer le daemon :**

```bash
ollama serve
Créer et activer un environnement Python virtuel :

```bash

python -m venv .venv
Sur Linux / macOS :

```bash

source .venv/bin/activate


Sur Windows PowerShell :

```bash
.venv\Scripts\Activate.ps1


Installer les dépendances :

```bash

pip install -r requirements.txt


Placer les fichiers PDF IPCC dans le dossier data/ :

```bash
data/
├── AR6_SYR_Full.pdf
├── AR6_SYR_SPM.pdf
├── WGI_SPM.pdf
🚀 Run
Ingestion des documents :

```bash

python ingest.py

Créer les embeddings et stocker dans la base vectorielle :

```bash

python embeddings.py

Démarrer l’API FastAPI :

```bash
uvicorn app:app --reload --port 8000

Lancer l’interface Streamlit :

```bash
streamlit run uistreamlit.py