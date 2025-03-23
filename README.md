# Renewable Energy Innovations - Article Summarization and Comparison

This project focuses on extracting key sentences from articles and comparing the similarities and differences between two articles on the topic of renewable energy. The approach utilizes pre-trained models like DistilBERT and Sentence Transformer for sentence embeddings and cosine similarity to identify key content and compare it across different articles.

## Key Features:
- **Text Preprocessing:** Cleans and formats the input article by removing unnecessary characters, hyperlinks, and sources.
- **Sentence Embedding:** Uses Sentence-Transformers' pre-trained DistilBERT model to generate sentence embeddings for key sentence extraction.
- **Key Sentence Extraction:** Extracts the top 5 most significant sentences from each article based on similarity.
- **Article Comparison:** Compares two articles by identifying the most similar and the most different key sentences.

## Installation

To run the project, you need Python 3.7+ and the following dependencies:

- PyTorch
- Transformers
- Sentence-Transformers
- NumPy
- Regex

### Steps to Install:

1. Install dependencies:
   ```bash
   pip install torch
   pip install transformers
   pip install sentence-transformers
   pip install numpy
2. Download the dataset

   Download the dataset `renewable-energy.zip` from [this link](https://storage.googleapis.com/ds--tasks-datasets/renewable-energy.zip) and store it in the `data/renewable-energy/` directory.

4. Make sure the articles (`.txt` files) are in the `data/renewable-energy/` folder.

## How to Run the Code

### Clean and Read Data

To clean and read the article text data, use the `read_data_into_list(articleId: str)` function to load the text file, clean it, and prepare it for further processing.

### Extract Key Sentences

Use the `extractKeySentences(articleId: str)` function to extract the top 5 key sentences for a given article.

### Compare Two Articles

Use the `compareArticles(articleId1: str, articleId2: str)` function to compare two articles and identify the most similar and most different key sentences.

### Example of Extracting Key Sentences:
```python
articleId = "1"  # Replace with the actual article ID
key_sentences = extractKeySentences(articleId)
print(key_sentences)
```

### Example of Comparing Articles:
```python
articleId1 = "1"  # First article ID
articleId2 = "2"  # Second article ID
comparison = compareArticles(articleId1, articleId2)
print(comparison)
```

### Project Explanation
Data Preprocessing:
The `clean_data(text)` function cleans the text by removing unwanted characters, hyperlinks, and sources.

Key Sentence Extraction:
The `extractKeySentences(articleId)` function reads the cleaned data and generates sentence embeddings using the pre-trained model. It then computes the average similarity for each sentence and ranks them based on importance.

Article Comparison:
The `compareArticles(articleId1, articleId2) function` compares the key sentences between two articles and identifies the most similar and the most different sentences by computing cosine similarity.

### Conclusion
The project allows for effective summarization and comparison of renewable energy-related articles, helping to analyze key insights and track thematic differences and similarities across multiple sources.

### Future Work
Include more advanced text summarization techniques to extract more context.

Extend article comparison to include a broader scope beyond just key sentences.

### License
This project is open-source and available under the MIT License.

### Acknowledgments
All the code in this project was written by me, but it was inspired by the reference sources listed in the code.
