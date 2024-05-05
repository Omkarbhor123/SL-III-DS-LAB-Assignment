import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

# 1. Extract Sample Document and Apply Document Preprocessing Methods

# Sample document
text = """
The quick brown fox jumps over the lazy dog. The dog is quite lazy and sleeps all day long.
"""

# Tokenization
tokens = nltk.word_tokenize(text)
print("Tokens:", tokens)

# POS Tagging
tagged = nltk.pos_tag(tokens)
print("\nPOS Tagged Tokens:", tagged)

# Stop words removal
stop_words = set(stopwords.words('english'))
filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
print("\nFiltered Tokens (Stop Words Removed):", filtered_tokens)

# Stemming
stemmer = PorterStemmer()
stemmed_tokens = [stemmer.stem(token) for token in filtered_tokens]
print("\nStemmed Tokens:", stemmed_tokens)

# Lemmatization
lemmatizer = WordNetLemmatizer()
lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
print("\nLemmatized Tokens:", lemmatized_tokens)

# 2. Create Representation of Documents by Calculating Term Frequency and Inverse Document Frequency

# Sample documents
documents = [
    "The quick brown fox jumps over the lazy dog.",
    "The dog is quite lazy and sleeps all day long.",
    "The fox is brown and quick, and it jumps over the dog."
]

# Create TF-IDF vectorizer
vectorizer = TfidfVectorizer()

# Fit and transform the documents to create a document-term matrix
X = vectorizer.fit_transform(documents)

# Get the feature names (terms)
terms = vectorizer.get_feature_names_out()

# Print the document-term matrix
print("\nDocument-Term Matrix:")
doc_term_df = pd.DataFrame(X.toarray(), columns=terms)
print(doc_term_df)

# Print the terms
print("\nTerms:")
print(terms)

# Get the term frequencies
term_frequencies_df = pd.DataFrame(term_frequencies, columns=terms)
print("\nTerm Frequencies:")
print(term_frequencies_df)

# Get the inverse document frequencies
idf_df = pd.DataFrame([idf], columns=terms)
print("\nInverse Document Frequencies:")
print(idf_df)
