from rank_bm25 import BM25Okapi
import nltk
from nltk.corpus import stopwords

# Download the stop word list
nltk.download('stopwords')

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

def main():
    # create a config dictionary
    config = {
        "path_to_stopwords": "stopwords.txt",
        "evs_max_evidences": 100,
        "qu": "sr",
        "sr_delimiter": ","
    }

    # create an instance of BM25Scoring
    bm25_scoring = BM25Scoring(config)

    # test input
    structured_representation = "What is the capital city of China?"
    evidences = [
        {"evidence_text": "Capital of the Philippines is a list of capital city."},
        {"evidence_text": "Beijing is the capital city of China."},
        {"evidence_text": "Tokyo is the capital city of Japan."}
    ]

    # get the top evidences
    scored_evidences = bm25_scoring.get_top_evidences(structured_representation, evidences)

    # print the result
    print(scored_evidences)

if __name__ == "__main__":
    main()