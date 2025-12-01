include .env

CHECK_DIRS := .

format:
	black $(CHECK_DIRS)


############### Offline Pipelines

run_api:
	uv run python -m library.api.api --port 8000 --host 0.0.0.0