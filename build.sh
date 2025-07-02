#!/bin/bash
apt-get update
apt-get install -y libpython3.13-dev build-essential libxml2-dev libxslt1-dev libmagic-dev
rm -rf /var/lib/apt/lists/*

# Install NLTK data
python -c "import nltk; nltk.download('punkt', download_dir='/opt/render/nltk_data')"
python -c "import nltk; nltk.download('wordnet', download_dir='/opt/render/nltk_data')"