from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer, DPRQuestionEncoder, DPRContextEncoder
import numpy as np

class HybridRetriever:
    def __init__(self, chunks):
        self.chunks = chunks
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.bm25 = BM25Okapi([self.tokenizer.encode(chunk, add_special_tokens=False) for chunk in chunks])
        self.question_encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
        self.context_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
    
    def hybrid_retrieval(self, query):
        # BM25 retrieval
        tokenized_query = self.tokenizer.encode(query, add_special_tokens=False)
        bm25_scores = self.bm25.get_scores(tokenized_query)

        # DPR retrieval
        question_embedding = self.question_encoder(self.tokenizer(query, return_tensors="pt")['input_ids']).pooler_output.detach().numpy()
        context_embeddings = np.array([self.context_encoder(self.tokenizer(chunk, return_tensors="pt")['input_ids']).pooler_output.detach().numpy() for chunk in self.chunks])
        dpr_scores = np.dot(context_embeddings, question_embedding.T).flatten()

        # Combine scores
        combined_scores = bm25_scores + dpr_scores
        top_k_indices = combined_scores.argsort()[-10:][::-1]
        return [self.chunks[i] for i in top_k_indices]

# Example usage
if __name__ == "__main__":
    with open('chunks.txt', 'r', encoding='utf-8') as file:
        chunks = [line.strip() for line in file.readlines()]
    retriever = HybridRetriever(chunks)
    query = "How to install CUDA?"
    results = retriever.hybrid_retrieval(query)
    for result in results:
        print(result)
