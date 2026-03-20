import spacy

nlp = spacy.load("en_core_web_sm")
from datetime import datetime
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")  

def extract_keywords_and_vectors(sentence, include_timestamp=True):
    doc = nlp(sentence)
    main_concepts = set()
    custom_excluded_words = {"me", "my", "i", "used", "when", "this", "parents", "bring", "sick", "got", "reminds"}
    relevant_entity_labels = ["PERSON", "NORP", "ORG", "GPE", "LOC", "PRODUCT", "EVENT", "DATE", "WORK_OF_ART", "LAW", "LANGUAGE"]

    # Named Entities
    extracted_entities = {ent.text.lower() for ent in doc.ents if ent.label_ in relevant_entity_labels}
    main_concepts.update(extracted_entities)

    # Noun Chunks
    for chunk in doc.noun_chunks:
        if any(not token.is_stop and not token.is_punct for token in chunk) and len(chunk.text.strip()) > 2:
            chunk_text_lower = chunk.text.lower()
            if len(chunk) == 1 and chunk.root.pos_ == "NOUN":
                main_concepts.add(chunk.root.lemma_.lower())
            else:
                main_concepts.add(chunk_text_lower)

    # Individual Nouns/Proper Nouns
    for token in doc:
        if token.pos_ in ["NOUN", "PROPN"] and not token.is_stop and not token.is_punct:
            lemma = token.lemma_.lower()
            if lemma not in custom_excluded_words and len(lemma.strip()) > 2:
                is_redundant = False
                for existing in main_concepts.copy():
                    if lemma == existing:
                        is_redundant = True
                        break
                    if len(existing.split()) > 1 and lemma in existing:
                        if lemma in [word.lemma_.lower() for word in nlp(existing) if word.pos_ in ["NOUN", "PROPN"]]:
                            is_redundant = True
                            break
                if not is_redundant:
                    main_concepts.add(lemma)

    # Filter substrings
    concepts_list = sorted(list(main_concepts), key=len, reverse=True)
    final_filtered_concepts = set()
    for i, concept_i in enumerate(concepts_list):
        is_substring = False
        for j, concept_j in enumerate(concepts_list):
            if i != j and concept_i in concept_j and len(concept_i.split()) < len(concept_j.split()):
                is_substring = True
                break
        if not is_substring:
            final_filtered_concepts.add(concept_i)

    # Final cleanup
    final_concepts = [c for c in final_filtered_concepts if c not in custom_excluded_words and len(c.strip()) > 2]
    if not final_concepts:
        return []

    # Embedding
    vectors = model.encode(final_concepts)

    # Output format: list of dicts
    if include_timestamp:
        timestamp = datetime.now().isoformat()
        return [
            {"keyword": keyword, "embedding": vector.tolist(), "timestamp": timestamp}
            for keyword, vector in zip(final_concepts, vectors)
        ]
    else:
        return [
            {"keyword": keyword, "embedding": vector.tolist()}
            for keyword, vector in zip(final_concepts, vectors)
        ]
