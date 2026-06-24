# ILSA Survey Analysis

A comprehensive collection of structured metadata from International Large-Scale Assessment (ILSA) research articles using machine learning methods. This repository contains extracted metadata from 130+ academic papers analyzing PISA, TIMSS, PIRLS, and other ILSA datasets.

## Dataset Overview

The `ilsa_survey_articles` directory contains:
- **JSON files**: Structured metadata for 130+ research articles (extracted using AI/LLM pipeline)
- **Database**: SQLite database (`ilsa_knowledge_base.db`) with all metadata
- **Parquet file**: Tabular dataset (`ilsa_master.parquet`) for analysis

## Data Structure

Each JSON file follows the `ILSAArticleMetadata` schema with two main sections:

### 1. Metadata Block
- `file_name`: Source PDF filename
- `title`: Article title
- `authors`: List of authors
- `year`: Publication year
- `doi`: Digital Object Identifier
- `venue`: Journal/conference name
- `publication_type`: Journal, conference, etc.
- `open_access`: Accessibility status
- `source_category`: Research type

### 2. Data Block
- `survey_design`: Weighting and sampling methodology
- `sample_details`: Sample size and country breakdown
- `ml_techniques`: Machine learning algorithms used
- `confounders_identified`: Predictor variables (13 categories)
- `main_findings`: Structured results with performance metrics
- `outcome_summary`: Narrative summary of findings

## Usage

### Python Analysis
```python
import pandas as pd
import json
from pathlib import Path

# Load all JSON files
json_dir = Path("ilsa_survey_articles/json")
articles = []
for json_file in json_dir.glob("*.json"):
    with open(json_file) as f:
        articles.append(json.load(f))

# Convert to DataFrame
df = pd.json_normalize(articles)
```

### SQLite Database
```python
import sqlite3

conn = sqlite3.connect("ilsa_survey_articles/ilsa_knowledge_base.db")
cursor = conn.cursor()

# List tables
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()
print(tables)
```

### Parquet File
```python
df = pd.read_parquet("ilsa_survey_articles/ilsa_master.parquet")
```

## Extraction Pipeline

The metadata was extracted using a custom pipeline:
1. **PDF Processing**: PyMuPDF for text extraction
2. **LLM Extraction**: OpenAI models for structured JSON extraction
3. **Schema Validation**: Pydantic models for data quality
4. **Storage**: SQLite and Parquet for analysis

## Research Categories

Variables are categorized into 13 domains:
1. Socioeconomic (ESCS, HOMEPOS, wealth)
2. Demographic (gender, age, immigration)
3. Student attitude (self-efficacy, motivation)
4. Student behavior (study time, homework)
5. Teacher (qualifications, experience)
6. School (type, resources, climate)
7. ICT (resources, computer use)
8. Curriculum (type, instructional time)
9. Parent/home (involvement, environment)
10. Process data (aggregate task metrics)
11. Prior achievement (test scores, grades)
12. Peer effects (classroom climate)
13. System level (GDP, education expenditure)

## Citation

If you use this dataset, please cite the original research articles and acknowledge this collection.

## License

The metadata extraction is provided for research purposes. Original article copyrights remain with their respective publishers.