"""Run the RAG exercise with Qwen3-0.6B (downloaded through ModelScope)."""
import argparse
import json
from agent import PolicyAgent, QwenGenerator
from rag_tool import PolicyRetriever

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('question', nargs='?', default='What exact phone number should I call for the University IT Service Desk?')
    parser.add_argument('--model-path', help='Local Qwen3-0.6B directory; otherwise use ModelScope.')
    parser.add_argument('--retrieval-only', action='store_true', help='Inspect evidence only; does not run the model.')
    args = parser.parse_args()
    retriever = PolicyRetriever()
    if args.retrieval_only:
        result = {'question': args.question, 'evidence': retriever.retrieve(args.question), 'model_run': False}
    else:
        result = PolicyAgent(QwenGenerator(args.model_path), retriever).answer(args.question)
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
