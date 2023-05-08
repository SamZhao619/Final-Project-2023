# Import the necessary modules
from sentence_transformers import SentenceTransformer
import numpy as np
import json
import spacy

# Define the SBERT model
model = SentenceTransformer('paraphrase-mpnet-base-v2')

# Import the CLOCQInterfaceClient class
from CLOCQ_api_Client import CLOCQInterfaceClient

# Create a new instance of the CLOCQInterfaceClient class
clocq = CLOCQInterfaceClient(host="https://clocq.mpi-inf.mpg.de/api", port="443")

def extract_sentence(facts):
    labels = [fact['label'] for fact in facts]
    sentence = ' '.join(labels)
    return sentence


if __name__ == "__main__":
    # Load the JSON object containing the question and answer
    with open('train_test.json', 'r') as f:
        data = json.load(f)

    # Extract the question text and answer
    question = data[0]['Question']
    answer = data[0]['Answer'][0]['AnswerArgument']

    # Set the value of k
    k = 100

    # Use the get_search_space method of the CLOCQInterfaceClient class to extract the relevant entities from the question
    search_space = clocq.get_search_space(question)

    # Extract the entity IDs, labels, and scores from the search_space variable and store them as a list of tuples in the entities variable.
    entities = [(item['item']['id'], item['item']['label'], item['score']) for item in search_space['kb_item_tuple']]
    # print(entities)

    # Sort the entities list by score in descending order using the sorted method and a lambda function that extracts the score from each tuple.
    entities_sorted = sorted(entities, key=lambda x: x[2], reverse=True)
    # print(entities_sorted)

    # Extract the sorted entity IDs from the entities_sorted list and store them in a new list.
    entity_ids_sorted = [entity[0] for entity in entities_sorted]
    # print(entity_ids_sorted)
    #
    # Retrieve the 1-hop and 2-hop neighborhoods of the entities
    sentences = []
    for entity_id in entity_ids_sorted:
        neighborhood = clocq.get_neighborhood(entity_id)
        # print(neighborhood)
        for fact in neighborhood:
            sentence = extract_sentence(fact)
            sentences.append(sentence)

    # Print the extracted sentences
    # print(sentences)

    # Calculate the sentence embeddings for the question and the extracted sentences
    question_embedding = model.encode(question)
    sentence_embeddings = model.encode(sentences)

    # Calculate the cosine similarity between the question embedding and the sentence embeddings
    similarity_scores = np.dot(question_embedding, sentence_embeddings.T) / (
                np.linalg.norm(question_embedding) * np.linalg.norm(sentence_embeddings, axis=1))

    # Sort the sentences based on their similarity scores
    sentences_sorted = [sentence for _, sentence in sorted(zip(similarity_scores, sentences), reverse=True)]

    # Check if the answer appears in the sorted sentences
    answer_presence = [answer in sentence for sentence in sentences_sorted[:k]]

    # Count the number of true values in the first k sentences
    num_true = answer_presence.count(True)

    # Print the sorted sentences and the number of true values
    print(sentences_sorted)
    print('Number of true values in first', k, 'sentences:', num_true)
    k_sentence = sentences_sorted[:k]
    print(k_sentence)
