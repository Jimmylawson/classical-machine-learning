import re
from typing import List
import numpy as np
from collections import Counter


class BagOfWordsVectorized:
    def __init__(self, max_features=5000):
        self.max_features = max_features
        self.words_to_index = None
        self.vocabulary = None

    def tokenize(self,message:str)->List[str]:
        message = message.lower()

        return re.findall(r"\b[a-z0-9']+\b",message)

    def build_vocabulary(self, messages):
        document_counts = Counter()
        for message in messages:
            tokens = set(self.tokenize(message))
            document_counts.update(tokens)
        most_common = document_counts.most_common(
            self.max_features
        )

        self.vocabulary  = [
            word for word, _ in most_common
        ]

        self.words_to_index = {
            word: index for index, word in enumerate(self.vocabulary)
        } # indexes determine where each word appears in a feature vector

        return self



    def vectorize_message(self, message):
        if self.words_to_index is None:
            raise ValueError(
                "Vocabulary not built. Call build_vocabulary first.")

        vector = np.zeros(len(self.words_to_index),
                          dtype=np.int8)
        tokens  = set(self.tokenize(message))

        for word in tokens:
            index  = self.words_to_index.get(word)
            if index is not None:
                vector[index] = 1

        return vector

    def transform(self,messages):
        vectors = [
            self.vectorize_message(message)
            for message in messages
        ]

        return np.vstack(vectors) # means vertical stack means place each vector on a separate row

    def fit_transform(self,message):
        self.build_vocabulary(message)
        return self.transform(message)

