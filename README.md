# Semantic Search for CSV

Search your Productboard CSV exports by meaning, not just keywords. Uses local AI embeddings to find semantically similar content - no API keys required.

## Quick Start

```bash
pip3 install -r requirements.txt
python3 app.py
```

Open http://localhost:8000 in your browser.

## How It Works

1. **Upload** a CSV file (up to 50 MB)
2. **Search** using natural language (e.g., "security concerns", "mobile performance issues")
3. **Filter** results by any column (status, priority, customer, etc.)
4. **Export** filtered results back to CSV

The tool automatically detects text columns, generates vector embeddings using `all-MiniLM-L6-v2` (runs locally), and ranks results by semantic similarity.

## Features

- **Semantic search**: "security concerns" matches "authentication issues", "data privacy", "unauthorized access"
- **Column-specific search**: Restrict search to specific columns (e.g., only Description)
- **Filters**: Combine semantic search with exact filters (equals, contains, greater/less than)
- **Similarity scores**: Each result shows a 0-100% relevance score
- **CSV export**: Export search results with similarity scores
- **Auto-detection**: Handles column types, delimiters, and encodings automatically

## Requirements

- Python 3.9+
- ~500 MB disk space for the embedding model (downloaded on first run)
