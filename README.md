# End-to-End Legal Chatbot

## Project Overview
This is a legal chatbot application built using LangChain, Flask, and Meta Llama2, with Qdrant as the vector database for semantic search.

## Prerequisites
- Conda (Anaconda or Miniconda)
- Python 3.10
- Hugging Face account (for downloading the Llama 2 model)

## Setup and Installation

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/end-to-end-legal-chatbot.git
cd end-to-end-legal-chatbot
```

### 2. Create Conda Environment
```bash
conda create -n lchatbot python=3.10 -y
conda activate lchatbot
```

### 3. Install Requirements
```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables
Create a `.env` file in the root directory with your Qdrant credentials:
```ini
api_key = "your_qdrant_api_key"
url_Qdrant = "your_qdrant_cluster_url"
```

### 5. Download Llama 2 Model
1. Visit: https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/tree/main
2. Download: `llama-2-7b-chat.ggmlv3.q4_0.bin`
3. Place the model in the `model/` directory

### 6. Prepare PDF Documents
- Place your legal PDF documents in the `Data/` directory
- Ensure PDFs are relevant to the legal domain you're focusing on

### 7. Initialize Vector Database
```bash
python store_index.py
```
This script will:
- Load PDFs from the `Data/` directory
- Split documents into chunks
- Generate embeddings
- Store embeddings in Qdrant

### 8. Run the Application
```bash
python app.py
```

### 9. Access the Chatbot
Open a web browser and navigate to:
```
http://localhost:8080
```

## Tech Stack
- **Language**: Python 3.10
- **Web Framework**: Flask
- **LLM**: Meta Llama2 (7B Chat)
- **Vector Database**: Qdrant
- **Embedding Model**: Sentence Transformers (all-MiniLM-L6-v2)
- **Libraries**: 
  - LangChain
  - Hugging Face Transformers
  - PyTorch


## Contributing
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License
Distributed under the MIT License. See `LICENSE` for more information.

