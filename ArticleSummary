import json
import torch
from transformers import DistilBertTokenizer, DistilBertModel
from sentence_transformers import SentenceTransformer, util
import numpy as np
import re

# Clean up the data by removing special characters, whitespace, hyperlinks, and sources
def clean_data(text):
    """
    This function cleans the raw article text by removing special characters, 
    extra whitespace, hyperlinks, and any "Sources" section typically at the end of articles.
    """
    # Remove unwanted symbols, new lines, and markdown characters
    text = text.replace("\n", "").replace("##", "").replace("*", "").strip()
    
    # remove hyperlinks of the form [text](url) from the article.
    pattern = r'\[[^\]]+\]\(([^)]+)\)'  
    text_without_link = re.sub(pattern, '', text)  

    # Remove any "Sources" section from the text (often found at the end of articles)
    text_without_sources = re.sub(r'Sources:.*', "", text_without_link)
    
    return text_without_sources

# Read the data from a file and clean the text before splitting it into sentences
def read_data_into_list(articleId: str):
    """
    This function reads an article from a file, cleans the text, and splits it into sentences.
    It returns a list of cleaned sentences from the article.
    """
    # Open the file containing the article and read its content
    with open(f"data/renewable-energy/{articleId}.txt", "r", encoding="utf-8") as file:
        first_line = file.readline()  # Read the first line (title or header)
        lines = file.readlines()  # Read the rest of the lines
    
    # Combine first line with the rest and clean the full article
    full_article = "".join(first_line) + "." + "".join(lines)
    
    # Split the article into sentences and clean each sentence
    article = clean_data(full_article).split(".")
    clean_article = [i.strip() for i in article if i != ""]  # Remove any empty entries
    return clean_article

# Load the pre-trained SentenceTransformer model
# **DistilBERT Selection**: 
# DistilBERT is a distilled version of BERT (Bidirectional Encoder Representations from Transformers).
# It retains 97% of BERT's performance while being 60% faster and smaller, making it ideal for tasks
# requiring efficient computation and fast inference, such as extracting key sentences from text.
# By using the 'msmarco-distilbert-base-tas-b' model, we're leveraging a version of DistilBERT fine-tuned model sentence transformer model
# Reference: 
# [DistilBERT Paper](https://arxiv.org/abs/1910.01108)
# [Sentence-Transformers GitHub](https://github.com/UKPLab/sentence-transformers)

model = SentenceTransformer('sentence-transformers/msmarco-distilbert-base-tas-b')

# Extract the top 5 key sentences based on their importance
def extractKeySentences(articleId: str):
    """
    This function extracts the top 5 key sentences from an article by generating
    sentence embeddings and calculating similarity scores between them.
    """
    # Step 1: Read and clean article data
    sentences = read_data_into_list(articleId)
    
    # Step 2: Generate sentence embeddings for each sentence
    embeddings = model.encode(sentences)  
    
    # Step 3: Generate a similarity matrix between each pair of sentences
    sim = np.zeros((len(sentences), len(sentences)))
    for i in range(len(sentences)):
        sim[i:, i] = util.cos_sim(embeddings[i], embeddings[i:])
    
    # Step 4: Make the matrix symmetric and calculate average similarity scores
    sim += sim.T - np.diag(sim.diagonal()) 
    mean_matrix = np.mean(sim, axis=0)  
    
    # Step 5: Sort sentences by mean similarity to find the top 5 key sentences
    desc_index = np.argsort(-mean_matrix)  
    key_sentences = [sentences[i-1] for i in desc_index[:5]]  
    
    # Return the results as a JSON-encoded string to be used for further process
    response = {
        "articleId": articleId,
        "keySentences": key_sentences
    }
    return json.dumps(response)

# Compare key sentences between two articles and find the most similar and most different sentences
def compareArticles(articleId1: str, articleId2: str):
    """
    This function compares the key sentences of two articles based on cosine similarity.
    It returns the most similar and most different sentences between the two articles.
    """
    # Step 1: Extract key sentences from both articles
    key_sentences1 = json.loads(extractKeySentences(articleId1))["keySentences"]
    key_sentences2 = json.loads(extractKeySentences(articleId2))["keySentences"]
    
    # Step 2: Generate embeddings for the key sentences
    embeddings1 = model.encode(key_sentences1) 
    embeddings2 = model.encode(key_sentences2)
    
    # Step 3: Calculate cosine similarity between each pair of key sentences
    sim = np.zeros((len(key_sentences1), len(key_sentences2)))
    for i in range(len(key_sentences1)):
        for j in range(len(key_sentences2)):
            similarity = util.cos_sim(embeddings1[i], embeddings2[j])
            sim[i, j] = similarity
    
    # Step 4: Identify the most similar and most different sentences based on cosine similarity
    max_index = np.unravel_index(np.argmax(sim), sim.shape)  # Index of most similar sentences
    similar_sentence = [key_sentences1[max_index[0]], key_sentences2[max_index[1]]]
    
    min_index = np.unravel_index(np.argmin(sim), sim.shape)  # Index of most different sentences
    different_sentence = [key_sentences1[min_index[0]], key_sentences2[min_index[1]]]
    
    # Return the comparison results as a JSON-encoded string
    response = {
        "articleId1": articleId1,
        "articleId2": articleId2,
        "MostSimilar": similar_sentence,
        "MostDifferent": different_sentence
    }
    return json.dumps(response)

# Can seperate the following tests into a separate script for future work

# Test data cleaning and extraction functionality
def test1():
    """
    This function tests the reading and cleaning process by extracting sentences
    from a given article. You can modify the articleId for testing.
    """
    articleId = "30"  # Modify this for different articles
    sentences = read_data_into_list(articleId)
    return sentences

# Test SentenceTransformer model and cosine similarity calculations
def test2():
    """
    This function tests the SentenceTransformer model by encoding sentences and
    calculating similarity between them using cosine similarity.
    """
    sentences = test1()
    embeddings = model.encode(sentences)  # Encode sentences into embeddings using DistilBERT
    print(embeddings.shape)  # Print the shape of the embeddings matrix
    
    # Generate a similarity matrix between all pairs of sentences
    sim = np.zeros((len(sentences), len(sentences)))
    for i in range(len(sentences)):
        sim[i:, i] = util.cos_sim(embeddings[i], embeddings[i:])
    
    # Symmetrize the matrix and calculate average similarity scores
    sim += sim.T - np.diag(sim.diagonal())
    mean_matrix = np.mean(sim, axis=0)
    print("Mean similarities:", mean_matrix)
    
    # Sort sentences by mean similarity and print the top 5
    desc_index = np.argsort(-mean_matrix)
    print("Top 5 sentences:", [sentences[i-1] for i in desc_index[:5]])


if __name__ == "__main__":
    # Run the following code to execute tests

    print("Testing article data cleaning...")
    test1_result = test1()
    print(test1_result)
    
    print("\nTesting sentence embeddings and similarity calculation...")
    test2()
