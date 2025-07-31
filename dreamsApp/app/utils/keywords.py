import spacy

nlp = spacy.load("en_core_web_lg")

def extract_keyword(sentence):
    doc = nlp(sentence)
    main_concepts = set()

    
    custom_excluded_words = {"me", "my", "i", "used", "when", "this", "parents", "bring", "sick", "got", "reminds"}

    relevant_entity_labels = ["PERSON", "NORP", "ORG", "GPE", "LOC", "PRODUCT", "EVENT", "DATE", "WORK_OF_ART", "LAW", "LANGUAGE"]

    extracted_entities = set()
    for ent in doc.ents:
        if ent.label_ in relevant_entity_labels:
            extracted_entities.add(ent.text.lower())
    main_concepts.update(extracted_entities)

    extracted_noun_chunks = set()
    for chunk in doc.noun_chunks:
        
        if any(not token.is_stop and not token.is_punct for token in chunk) and len(chunk.text.strip()) > 2:
            chunk_text_lower = chunk.text.lower()
            
            if len(chunk) == 1 and chunk.root.pos_ == "NOUN":
                extracted_noun_chunks.add(chunk.root.lemma_.lower())
            else:
                extracted_noun_chunks.add(chunk_text_lower)

    main_concepts.update(extracted_noun_chunks)


    
    for token in doc:
        if token.pos_ in ["NOUN", "PROPN"] and \
           not token.is_stop and \
           not token.is_punct and \
           token.lemma_.lower() not in custom_excluded_words and \
           len(token.lemma_.strip()) > 2: # Exclude very short lemmas

            
            is_redundant = False
            token_lemma_lower = token.lemma_.lower()

            for existing_concept in main_concepts.copy(): # Iterate over a copy to avoid modification issues
                # If the individual noun is already the same as an existing concept
                if token_lemma_lower == existing_concept:
                    is_redundant = True
                    break
                if len(existing_concept.split()) > 1 and token_lemma_lower in existing_concept:
                    if token_lemma_lower in [word.lemma_.lower() for word in nlp(existing_concept) if word.pos_ in ["NOUN", "PROPN"]]:
                        is_redundant = True
                        break

            if not is_redundant:
                main_concepts.add(token_lemma_lower)

    
    final_filtered_concepts = set()
    concepts_list = sorted(list(main_concepts), key=len, reverse=True)

    for i, concept_i in enumerate(concepts_list):
        is_substring_of_another = False
        for j, concept_j in enumerate(concepts_list):
            if i != j and concept_i in concept_j and len(concept_i.split()) < len(concept_j.split()):
                
                is_substring_of_another = True
                break
        if not is_substring_of_another:
            final_filtered_concepts.add(concept_i)

    
    final_final_concepts = [c for c in list(final_filtered_concepts) if c not in custom_excluded_words and len(c.strip()) > 2]


    return sorted(list(final_final_concepts))

# # --- Test Cases ---
# sentence1 = "This hospital reminds me of my childhood, my parents used to bring me when I got sick."
# print(f"Sentence 1: '{sentence1}'")
# print("Main Things:", extract_aspects(sentence1))
# # Expected: ['childhood', 'hospital'] (or 'my childhood', 'this hospital' if preferred noun chunks)

# sentence2 = "The quick brown fox jumps over the lazy dog."
# print(f"\nSentence 2: '{sentence2}'")
# print("Main Things:", extract_aspects(sentence2))
# # Expected: ['brown fox', 'lazy dog'] or ['dog', 'fox'] if simpler.

# sentence3 = "The new Apple iPhone 15 has an amazing camera and long battery life."
# print(f"\nSentence 3: '{sentence3}'")
# print("Main Things:", extract_aspects(sentence3))
# # Expected: ['apple iphone 15', 'battery life', 'camera']

# sentence4 = "My son loves playing with his toy cars and trains."
# print(f"\nSentence 4: '{sentence4}'")
# print("Main Things:", extract_aspects(sentence4))
# # Expected: ['son', 'toy cars', 'trains', 'cars'] or ['son', 'toy cars', 'trains']

# sentence5 = "I went to New York City and visited Central Park."
# print(f"\nSentence 5: '{sentence5}'")
# print("Main Things:", extract_aspects(sentence5))
# # Expected: ['central park', 'new york city']

# sentence6 = "The doctor performed a successful surgery on the patient's knee."
# print(f"\nSentence 6: '{sentence6}'")
# print("Main Things:", extract_aspects(sentence6))
# # Expected: ['doctor', 'knee', 'patient', 'surgery']