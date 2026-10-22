export B=${B:?set B to your workspace directory}
export HD_CODE=${HD_CODE:-$B/code}
source "$B/env/${1:-venv}/bin/activate"
export HF_HOME=$B/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export DJINN_PATH=$B/repos_djinn CATCH_PATH=$B/repos_catch
export PYTHONPATH=$HD_CODE/src${PYTHONPATH:+:$PYTHONPATH}
export MODEL=${MODEL:-$B/models/Qwen3-4B}
export TASKS=${TASKS:-$HD_CODE/tasks_out/tasks.jsonl}
