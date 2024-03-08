# GenerativeAI_Article_Summary
* This project uses Distilbert and sentence transformer to generate article summary.
* In the project, the key objectives are to extract the key sentences from each article and also compare the similaries and differences between two articles.
* The articles can be any text file with header and hyperlink. This project uses sample text from a data folder consisting of articles describing renewable energy.
* Cosine similarity matrix is computed and used as the matrix for the importance of a senetence among the article.
* According to the definition of sentence transformer, senetence embeddings calculation is optimized in the [sentence transformer models](https://huggingface.co/sentence-transformers/msmarco-distilbert-base-tas-b).
  
