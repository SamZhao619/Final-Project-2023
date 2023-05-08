from sentence_transformers import SentenceTransformer
import numpy as np
import json
from rank_bm25 import BM25Okapi
import nltk
from nltk.corpus import stopwords

# Download the stop word list
nltk.download('stopwords')


# Import the CLOCQInterfaceClient class
from CLOCQ_api_Client import CLOCQInterfaceClient

# Create a new instance of the CLOCQInterfaceClient class
clocq = CLOCQInterfaceClient(host="https://clocq.mpi-inf.mpg.de/api", port="443")

def extract_sentence(facts):
    labels = [fact['label'] for fact in facts]
    sentence = {'evidence_text': ' '.join(labels)}
    return sentence

class BM25Scoring:
    def __init__(self, config):
        self.stopwords = set(stopwords.words('english'))

        self.max_evidences = config["evs_max_evidences"]
        if config["qu"] == "sr":
            self.sr_delimiter = config["sr_delimiter"].strip()
        else:
            self.sr_delimiter = " "

    def get_top_evidences(self, structured_representation, evidences):
        """
        Retrieve the top-100 evidences among the retrieved ones,
        for the given AR.
        """

        def tokenize(string):
            """Function to tokenize string (word-level)."""
            string = string.replace(",", " ")
            string = string.replace(self.sr_delimiter, " ")
            string = string.strip()
            return [word.lower() for word in string.split() if not word in self.stopwords]

        if not evidences:
            return evidences

        # tokenize
        mapping = {
            " ".join(tokenize(evidence["evidence_text"])): evidence for evidence in evidences
        }
        tokenized_sr = tokenize(structured_representation)

        # create corpus
        tokenized_corpus = [tokenize(evidence["evidence_text"]) for evidence in evidences]
        bm25_module = BM25Okapi(tokenized_corpus)

        # scoring
        scores = bm25_module.get_scores(tokenized_sr)

        # retrieve top-k
        ranked_indices = sorted(
            range(len(tokenized_corpus)), key=lambda i: scores[i], reverse=True
        )[: self.max_evidences]

        scored_evidences = [
            mapping[" ".join(tokenized_corpus[index])] for i, index in enumerate(ranked_indices)
        ]
        return scored_evidences

def bm25(question, sentences, k):
    config = {
        "path_to_stopwords": "stopwords.txt",
        "evs_max_evidences": 1000,
        "qu": "sr",
        "sr_delimiter": ","
    }

    # create an instance of BM25Scoring
    bm25_scoring = BM25Scoring(config)

    # test input
    structured_representation = question
    evidences = sentences

    # get the top evidences
    scored_evidences = bm25_scoring.get_top_evidences(structured_representation, evidences)

    # Return the first k scored evidences based on the value of k
    return scored_evidences[:k]

def sbert(question, sentences, k):
    model = SentenceTransformer('paraphrase-mpnet-base-v2')

    # Calculate the sentence embeddings for the question and the extracted sentences
    question_embedding = model.encode(question)
    sentence_embeddings = model.encode(sentences)

    # Calculate the cosine similarity between the question embedding and the sentence embeddings
    similarity_scores = np.dot(question_embedding, sentence_embeddings.T) / (
            np.linalg.norm(question_embedding) * np.linalg.norm(sentence_embeddings, axis=1))

    # Sort the sentences based on their similarity scores
    sentences_sorted = [sentence for _, sentence in sorted(zip(similarity_scores, sentences), reverse=True)]

    # Return the first k sentences based on the value of k
    return sentences_sorted[:k]


if __name__ == "__main__":
    # Load the JSON object containing the question and answer
    with open('train_test.json', 'r') as f:
        data = json.load(f)

    # Extract the question text and answer
    question = data[0]['Question']
    answer = data[0]['Answer'][0]['AnswerArgument']

    # Set the maximum value of k
    k = 100

    # Use the get_search_space method of the CLOCQInterfaceClient class to extract the relevant entities from the question
    search_space = clocq.get_search_space(question)
    # print(search_space)

    # Extract the entity IDs, labels, and scores from the search_space variable and store them as a list of tuples in the entities variable.
    entities = [(item['item']['id'], item['item']['label'], item['score']) for item in search_space['kb_item_tuple']]
    # print(entities)

    # Sort the entities list by score in descending order using the sorted method and a lambda function that extracts the score from each tuple.
    entities_sorted = sorted(entities, key=lambda x: x[2], reverse=True)
    # print(entities_sorted)

    # Extract the sorted entity IDs from the entities_sorted list and store them in a new list.
    entity_ids_sorted = [entity[0] for entity in entities_sorted]
    # print(entity_ids_sorted)

    # Retrieve the 1-hop and 2-hop neighborhoods of the entities
    sentences = []
    for entity_id in entity_ids_sorted:
        neighborhood = clocq.get_neighborhood(entity_id)
        for fact in neighborhood:
            sentence = extract_sentence(fact)
            sentences.append(sentence)

    bm25(question, sentences, k)
    sbert(question, sentences, k)



