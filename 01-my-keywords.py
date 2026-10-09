#Source: https://towardsdatascience.com/keyword-extraction-with-bert-724efca412ea

#pip install sentence-transformers

from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer

n_gram_range = (1, 1)
stop_words = "english"

text = """


Hamilton
1 dead, 3 others sent to hospital, including 1 with critical injuries, following head-on crash in Hamilton
By Joanna Lavoie

Published: October 06, 2026 at 11:07a.m. EDT
An Ontario Provincial Police vehicle is pictured in this file image. (File photo/Supplied/Ontario Provincial Police | Photo credit: Bobb Barratt)

One person has been killed in a head-on collision in Hamilton that sent three others to the hospital, including one with critical injuries, Ontario Provincial Police (IPP) say.

The crash happened in the town of Dundas, Ont., on Tuesday shortly before 7:30 a.m. on Highway 5, just east of Highway 8.

The two other patients are in non-life-threatening condition, police said.

The OPP’s Traffic Support Unit is now on scene investigating.

Motorists are being advised to expect a “lengthy” closure on Highway 5 between Highway 8 and Middletown Road, notably between the roundabout at Peter’s Corners and Middletown Road.

The closure is expected to last around three hours, until mid-afternoon, police said.

Drivers should avoid that area, they said.



 """

# Extract candidate words/phrases
count = CountVectorizer(ngram_range=n_gram_range, stop_words=stop_words).fit([text])
candidates = count.get_feature_names_out()

# print ("candidates: ", candidates)

model = SentenceTransformer('distilbert-base-nli-mean-tokens')
doc_embedding = model.encode([text])
candidate_embeddings = model.encode(candidates)

print ("candidate_embeddings: ", candidate_embeddings)


from sklearn.metrics.pairwise import cosine_similarity

top_n = 7
distances = cosine_similarity(doc_embedding, candidate_embeddings)
keywords = [candidates[index] for index in distances.argsort()[0][-top_n:]]

print(keywords)

# Replace your current import:
from transformers import is_torch_npu_available

try:
    # Works in some transformers versions
    from transformers import is_torch_npu_available  # type: ignore
except ImportError:
    try:
        # Works in versions where the symbol is not re-exported at top level
        from transformers.utils import is_torch_npu_available  # type: ignore
    except ImportError:
        # Safe fallback for environments without NPU support helpers
        def is_torch_npu_available() -> bool:
            return False


# Optional quick check:
if __name__ == "__main__":
    import transformers
    print("transformers version:", transformers.__version__)
    print("NPU available:", is_torch_npu_available())