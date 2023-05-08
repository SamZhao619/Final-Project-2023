from sentence_transformers import SentenceTransformer
import numpy as np
import json


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

# Load the JSON object containing the test dataset
with open('train_test.json', 'r') as f:
    test_data = json.load(f)

# Initialize the MRR, compression, and recall scores
total_reciprocal_rank = 0
total_compression = 0
total_correct_answers = 0

# Loop over each example in the test dataset
for example in test_data:
    # Extract the question text and answer for the current example
    question = example['Question']
    answer = example['Answer'][0]['AnswerArgument']

    # Set the maximum value of k
    max_k = 100

    # Use the get_search_space method of the CLOCQInterfaceClient class to extract the relevant entities from the question
    search_space = clocq.get_search_space(question)

    # Extract the entity IDs, labels, and scores from the search_space variable and store them as a list of tuples in the entities variable.
    entities = [(item['item']['id'], item['item']['label'], item['score']) for item in search_space['kb_item_tuple']]

    # Sort the entities list by score in descending order using the sorted method and a lambda function that extracts the score from each tuple.
    entities_sorted = sorted(entities, key=lambda x: x[2], reverse=True)

    # Extract the sorted entity IDs from the entities_sorted list and store them in a new list.
    entity_ids_sorted = [entity[0] for entity in entities_sorted]

    # Retrieve the 1-hop and 2-hop neighborhoods of the entities
    sentences = []
    for entity_id in entity_ids_sorted:
        neighborhood = clocq.get_neighborhood(entity_id)
        for fact in neighborhood:
            sentence = extract_sentence(fact)
            sentences.append(sentence)

    # Calculate the sentence embeddings for the question and the extracted sentences
    question_embedding = model.encode(question)
    sentence_embeddings = model.encode(sentences)

    # Calculate the cosine similarity between the question embedding and the sentence embeddings
    similarity_scores = np.dot(question_embedding, sentence_embeddings.T) / (
                np.linalg.norm(question_embedding) * np.linalg.norm(sentence_embeddings, axis=1))

    # Sort the sentences based on their similarity scores
    sentences_sorted = [sentence for _, sentence in sorted(zip(similarity_scores, sentences), reverse=True)]

    # Determine the value of k that maximizes the ratio of num_true / k
    max_ratio = 0
    best_k = 0
    for k in range(1, max_k + 1):
        answer_presence = [answer in sentence for sentence in sentences_sorted[:k]]
        num_true = answer_presence.count(True)
        ratio = num_true / k
        if ratio > max_ratio:
            max_ratio = ratio
            best_k = k

    # Calculate the MRR score for the current example
    reciprocal_rank = 0
    for rank, sentence in enumerate(sentences_sorted):
        if answer in sentence:
            reciprocal_rank = 1 / (rank + 1)
            break
    total_reciprocal_rank += reciprocal_rank

    # Calculate the compression score for the current example
    compression = 0
    for sentence in sentences_sorted:
        compression += 1
        if answer in sentence:
            break
    total_compression += compression

    # Calculate the recall score for the current example
    if answer in sentences_sorted[:best_k]:
        total_correct_answers += 1

# Calculate the average MRR score, compression score, and recall score over all examples in the test dataset
num_examples = len(test_data)
mrr_score = total_reciprocal_rank / num_examples
compression_score = total_compression / num_examples
recall_score = total_correct_answers / num_examples

# Print the MRR score, compression score, and recall score
print('MRR score:', mrr_score)
print('Compression score:', compression_score)
print('Recall score:', recall_score)