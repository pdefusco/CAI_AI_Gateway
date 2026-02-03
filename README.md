# AI Gateway in Cloudera AI

## Usage

```
curl -X POST https://ai.ml-fadd6e9b-75f.pdf-feb.a465-9q4k.cloudera.site/model-a \
  -H "Content-Type: application/json" \
  -d '{
        "inputs": "What is a finite state machine?"
      }'

curl -X POST http://localhost:8000/inference \
  -H "Content-Type: application/json" \
  -d '{
        "model_name": "model-b",
        "inputs": "Summarize this text"
      }'
```
