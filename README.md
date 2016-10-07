# Information Retrieval using Map-Reduce

Information retrieval (IR) is concerned with finding material (e.g., documents) of an unstructured nature (usually text) in response to an information need (e.g., a query) from large collections. My approach to identify relevant documents is to compute scores based on the matches between terms in the query and terms in the documents. For example, a document with words such as ball , team, score, championship is likely to be about sports. It is helpful to define a weight for each term in a document that can be meaningful for computing such a score. We describe below popular information retrieval metrics such as term frequency, inverse document frequency, and their product, term frequency‐inverse document frequency (TF‐IDF), that are used to define weights for terms. 

The job accepts as input a user query and outputs a list of documents with scores that best matches the query.

