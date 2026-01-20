apt install curl
sudo apt-get install zstd
curl -fsSL https://ollama.com/install.sh | sh
ollama serve &
ollama pull llama3.2:3b
pip3 install "numpy<2.0"
pip3 install --use-deprecated=legacy-resolver sentence-transformers
pip3 install "huggingface-hub>=0.34.0,<1.0"
pip3 install einops
python3 -c "import chromadb, sentence_transformers, numpy, torch; print('✓ All packages imported successfully')"
chmod +x setup.py ingest.py query.py
