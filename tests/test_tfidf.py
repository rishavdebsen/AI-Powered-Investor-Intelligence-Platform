from investor_intel.embeddings.local_tfidf import TfidfEmbedder, cosine


def test_relevant_text_scores_higher_than_unrelated():
    docs = [
        "Revenue grew 11% year over year and operating margin was 23%.",
        "Net income was $742 million with diluted EPS of $5.16.",
        "The recipe calls for basil, tomatoes and olive oil.",
    ]
    emb = TfidfEmbedder()
    emb.fit(docs)
    q = emb.embed("revenue growth and margin")
    assert cosine(q, emb.embed(docs[0])) > cosine(q, emb.embed(docs[2]))


def test_empty_embedding_is_zero():
    assert cosine({}, {"a": 1.0}) == 0.0
