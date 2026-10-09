"""Retrieve policy sentences from the supplied, unchanged document loader."""
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from policy_loader import load_policy_documents

UNKNOWN = "I don't know the University IT Service Desk phone number because it is not specified in the provided policies."
PHONE_QUESTION = re.compile(r"\b(phone|telephone|hotline|number.*(?:call|dial)|(?:call|dial).*number)\b|电话号码|联系电话", re.I)
PHONE_EVIDENCE = re.compile(r"\b(?:phone|telephone|tel|hotline|call|dial)\b[^\n.!?]{0,100}\+?\d[\d ()-]{5,}\d", re.I)

class PolicyRetriever:
    def __init__(self):
        documents = load_policy_documents()
        self.documents = documents
        self.sentences = []
        seen = set()
        for doc in documents:
            # Titles are not evidence; retain actual policy sentences and provenance.
            body = doc.page_content.split('\n\n', 1)[-1]
            for sentence in re.split(r'(?<=[.!?])\s+', body):
                if sentence and sentence not in seen:
                    seen.add(sentence)
                    self.sentences.append({'text': sentence, 'policy_id': doc.metadata['policy_id']})
        self.vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
        self.matrix = self.vectorizer.fit_transform(s['text'] for s in self.sentences)

    def retrieve(self, question, k=6):
        if not question.strip():
            raise ValueError('Question must not be empty.')
        if k < 1:
            raise ValueError('k must be positive.')
        scores = cosine_similarity(self.vectorizer.transform([question]), self.matrix)[0]
        candidates = range(len(self.sentences))
        # Numeric facts require actual phone evidence, not unrelated 15/24-hour rules.
        if PHONE_QUESTION.search(question):
            candidates = [i for i in candidates if PHONE_EVIDENCE.search(self.sentences[i]['text'])]
        ranked = sorted(candidates, key=lambda i: (-scores[i], i))
        return [dict(self.sentences[i], id=f'E{n + 1}', score=float(scores[i]))
                for n, i in enumerate([i for i in ranked if scores[i] > 0][:k])]


def rag_tool(question, retriever=None):
    return (retriever or PolicyRetriever()).retrieve(question)
