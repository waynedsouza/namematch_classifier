from sentence_transformers import SentenceTransformer

#model_id = "sentence-transformers/all-MiniLM-L6-v2"
model_id = "sentence-transformers/all-roberta-large-v1"
model = SentenceTransformer(model_id)