.PHONY: install lint format test eval run docker-build docker-run azure-deploy

PY ?= python
RG ?= rg-ragagent
LOCATION ?= eastus

install:
	$(PY) -m pip install -e ".[dev]"

lint:
	ruff check .
	ruff format --check .

format:
	ruff format .
	ruff check --fix .

test:
	$(PY) -m pytest -q

eval:
	ragagent eval --out reports/eval_results.json --min-hit-at-5 0.9
	ragagent eval --golden data/eval/golden_paraphrase.jsonl --out reports/eval_paraphrase.json

run:
	uvicorn ragagent.api:app --reload --port 8000

docker-build:
	docker build -t azure-rag-agent:local .

docker-run:
	docker run --rm -p 8000:8000 azure-rag-agent:local

# Not executed in this build (no Azure subscription). Requires `az login`.
azure-deploy:
	az group create -n $(RG) -l $(LOCATION)
	az deployment group create -g $(RG) -f infra/main.bicep -p infra/main.parameters.json
	ACR=$$(az deployment group show -g $(RG) -n main --query properties.outputs.acrName.value -o tsv); \
	APP=$$(az deployment group show -g $(RG) -n main --query properties.outputs.containerAppName.value -o tsv); \
	az acr build -r $$ACR -t azure-rag-agent:latest . && \
	az containerapp update -g $(RG) -n $$APP --image $$ACR.azurecr.io/azure-rag-agent:latest
