"""Central paths and constants for the OpenHA self-evolution framework.

All paths point at the existing /data3/openha_repro layout and the new
/data3/openha_evo workspace so the framework can run directly on the server.
"""

import os

# ---- Existing shared project (read-mostly) ----
OPENHA_REPRO = "/data3/openha_repro"
OPENHA_ROOT = os.path.join(OPENHA_REPRO, "OpenHA")
BASE_MODEL = os.path.join(OPENHA_REPRO, "models", "minecraft-openha-qwen2vl-7b-2509")
TEST_TASKS = "/data3/openha_evo/bench/test_v2_tasks.txt"
TRAIN_TASKS = "/data3/openha_evo/bench/train_v2_tasks.txt"
RUN_STREAMING_EVAL = os.path.join(OPENHA_REPRO, "run_streaming_eval_evo.py")
TRAIN_SCRIPT = os.path.join(OPENHA_REPRO, "train_openha_lora_sft.py")
BUILD_DATA_SCRIPT = os.path.join(OPENHA_REPRO, "build_openha_lora_sft_v3_data_evo.py")

# Existing v4 records used to seed the first Main-LLM planning round.
V4_SFT_RECORD = "/data3/openha_runs/zengqifen/streaming_eval_results/openha_sft_v4_120_v1/openha_eager_sft_v4"
V4_SFT_ICL_RECORD = "/data3/openha_runs/zengqifen/streaming_eval_results/openha_sft_v4_icl_120_v1/openha_eager_sft_v4_icl"

# ---- New evolution workspace ----
EVO_ROOT = "/data3/openha_evo"
DATA_DIR = os.path.join(EVO_ROOT, "data")
ICL_BANK_DIR = os.path.join(EVO_ROOT, "data", "icl_bank")
TASK_REGISTRY_PATH = os.path.join(DATA_DIR, "task_registry.json")
SFT_DATASETS_DIR = os.path.join(DATA_DIR, "sft_datasets")
ADAPTERS_DIR = os.path.join(EVO_ROOT, "adapters")
LOGS_DIR = os.path.join(EVO_ROOT, "logs")
PROMPTS_DIR = os.path.join(EVO_ROOT, "main_llm", "prompts")

# ---- zengqifen runtime (matching the working zengqifen_resume_sft_v4_eval.sh) ----
RUN_ROOT = "/data3/openha_runs/zengqifen"
PYTHON = "/opt/miniconda3/envs/vllm_env/bin/python"
SHARED_FLASH_ATTN = os.path.join(OPENHA_REPRO, "flash-attention-shared")
SHARED_MINE_ENGINE = os.path.join(OPENHA_REPRO, "MineStudio_shared", "engine")
HF_DATASETS_CACHE = os.path.join(RUN_ROOT, "cache", "hf_datasets_cache")

# ---- Rigorous deterministic benchmark (test-v2) ----
BENCH_SEED = 42
BENCH_DIR = os.path.join(EVO_ROOT, "bench")
BENCH_V2_TASKS = os.path.join(BENCH_DIR, "test_v2_tasks.txt")
BENCH_V2_TRAIN_TASKS = os.path.join(BENCH_DIR, "train_v2_tasks.txt")
BENCH_V2_MANIFEST = os.path.join(BENCH_DIR, "test_v2_manifest.json")

# ---- Runtime defaults ----
GPU_FREE_THRESHOLD_MB = 20000
DEFAULT_MAX_STEPS = 400
CATEGORIES = ("kill_entity", "mine_block", "custom", "craft_item")


def category(task: str) -> str:
    """Map a task name to its behaviour category."""
    for c in CATEGORIES:
        if task.startswith(c):
            return c
    if "interact_with" in task:
        return "custom"
    return "other"


def task_dirname(task: str) -> str:
    """Record directories use underscores instead of the ':' separator."""
    return task.replace(":", "_")


def ensure_dirs():
    for p in (DATA_DIR, ICL_BANK_DIR, SFT_DATASETS_DIR, ADAPTERS_DIR, LOGS_DIR, PROMPTS_DIR):
        os.makedirs(p, exist_ok=True)
