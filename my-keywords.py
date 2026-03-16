#Source: https://towardsdatascience.com/keyword-extraction-with-bert-724efca412ea

from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer

n_gram_range = (1, 1)
stop_words = "english"

text = """
Georgian College is a College of Applied Arts and Technology in Ontario, Canada. Georigan has 13,000 full-time students, including 4,500 international students from 85 countries, across seven campuses, the largest being in Barrie.
The college was established during the formation of Ontario's college system in 1967. Colleges of Applied Arts and Technology were established on May 21, 1965, when the Ontario system of public colleges was created. 
Georgian College offers academic upgrading, apprenticeship training, certificate, diploma, graduate certificate, college degree and university programs (including combined degree-diplomas) and part-time studies in such areas such as automotive business, business and management, community safety, computer studies, design and visual arts, engineering technology and environmental studies, health, wellness and sciences, hospitality, tourism and recreation, human services, Indigenous studies, liberal arts, marine studies, and skilled trades.
 """

# Extract candidate words/phrases
count = CountVectorizer(ngram_range=n_gram_range, stop_words=stop_words).fit([text])
candidates = count.get_feature_names_out()

print (candidates)

model = SentenceTransformer('distilbert-base-nli-mean-tokens')
doc_embedding = model.encode([text])
candidate_embeddings = model.encode(candidates)

print (candidate_embeddings)


from sklearn.metrics.pairwise import cosine_similarity

top_n = 7
distances = cosine_similarity(doc_embedding, candidate_embeddings)
keywords = [candidates[index] for index in distances.argsort()[0][-top_n:]]

print(keywords)