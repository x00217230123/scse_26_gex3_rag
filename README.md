# SCSE 26 GEX3 RAG

Uses Qwen3-0.6B and only the provided policy file. `policy_loader.py` is unchanged.
The loader supplies 1,196 policy records; TF-IDF retrieves unique policy sentences
with source IDs. Qwen selects directly relevant evidence IDs. The application returns
only selected source sentences, and rejects malformed or unknown IDs. Missing phone
facts are filtered before generation so unrelated time limits cannot become phone numbers.
This is an extractive RAG design: it favors supported text over free-form generation.
Model relevance selection can still fail or abstain on answerable questions.

## Windows PowerShell

```powershell
cd C:\Users\HUANG\Desktop\SCSE\scse_26_gex3_rag
uv venv
.\.venv\Scripts\Activate.ps1
uv pip install -r requirements.txt
python check_data.py
python -m unittest -v test_rag.py
python rag.py --model-path ..\SCSE_26_GEX3\models\qwen3-0.6b
```

Omit `--model-path` to download Qwen3-0.6B through ModelScope. Per the assignment,
turn off your VPN for the ModelScope download. No 8B model is used.

```powershell
python rag.py "Which wireless network should guests use?" --model-path ..\SCSE_26_GEX3\models\qwen3-0.6b
python rag.py --retrieval-only
```

The target phone question must return only:

> I don't know the University IT Service Desk phone number because it is not specified in the provided policies.

Tests inject model outputs to check retrieval and validation. They do not constitute
an actual Qwen inference run. A real run requires downloaded model weights and dependencies.
