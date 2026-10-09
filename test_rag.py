"""Regression tests for real policy retrieval and evidence validation, without model weights."""
import unittest
from agent import PolicyAgent
from rag_tool import PolicyRetriever, UNKNOWN

class GroundingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.retriever = PolicyRetriever()

    def test_policy_count(self):
        self.assertEqual(len(self.retriever.documents), 1196)

    def test_missing_phone_cannot_be_fabricated(self):
        for question in ['What exact phone number should I call for the University IT Service Desk?',
                         'What is the IT support telephone number?', 'University IT 联系电话是多少？']:
            with self.subTest(question=question):
                self.assertEqual(self.retriever.retrieve(question), [])
                agent = PolicyAgent(lambda q, e: '{"evidence_ids": ["E1"]}', self.retriever)
                self.assertEqual(agent.answer(question), {'answer': UNKNOWN, 'sources': []})

    def test_supported_wifi_answer(self):
        question = 'Which wireless network should guests use?'
        evidence = self.retriever.retrieve(question)
        target = next(e for e in evidence if 'Guests should use Campus-Guest' in e['text'])
        agent = PolicyAgent(lambda q, e: '{"evidence_ids": ["' + target['id'] + '"]}', self.retriever)
        answer = agent.answer(question)
        self.assertIn('Campus-Guest', answer['answer'])
        self.assertTrue(answer['sources'])

    def test_bad_model_outputs_abstain(self):
        for output in ['Call 123456789.', '{"evidence_ids": ["E999"]}', '{"evidence_ids": "E1"}',
                       '{"evidence_ids": [null]}', '{"evidence_ids": []}']:
            agent = PolicyAgent(lambda q, e, output=output: output, self.retriever)
            self.assertEqual(agent.answer('What is the help desk phone number?')['answer'], UNKNOWN)

    def test_unrelated_question(self):
        self.assertEqual(self.retriever.retrieve('Who won the intergalactic chess championship?'), [])

if __name__ == '__main__':
    unittest.main()
