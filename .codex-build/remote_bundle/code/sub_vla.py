"""Thin wrapper around the existing OpenHA sub-LLM tooling.

The framework does not reimplement inference/training; it shells out to the
proven scripts in /data3/openha_repro and just standardises the interface.
"""

import os
import subprocess

import config


def _env():
    env = os.environ.copy()
    env.update({
        "OPENHA_ROOT": config.OPENHA_ROOT,
        "HF_HOME": os.path.join(config.OPENHA_REPRO, "models", "hf_cache"),
        "HF_HUB_CACHE": os.path.join(config.OPENHA_REPRO, "models", "hf_cache"),
        "HF_DATASETS_CACHE": config.HF_DATASETS_CACHE,
        "TIMM_OFFLINE": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    })
    # flash_attn is a setup.py develop install under /home/user (unreadable),
    # so point at the shared copy.
    env["PYTHONPATH"] = config.SHARED_FLASH_ATTN + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    return env


def run_eval(config_name, task_list, record_path, gpu_id, max_steps=None, log_file=None,
             lora_adapter_path=None, bench_seed=None, extra_env=None):
    """Launch one eval config. Returns the subprocess handle (blocking)."""
    max_steps = max_steps or config.DEFAULT_MAX_STEPS

    record_name = os.path.basename(record_path.rstrip("/")) or config_name
    mine_cache = os.path.join(config.RUN_ROOT, "cache", record_name)
    tmp_dir = os.path.join(config.RUN_ROOT, "tmp", record_name)
    java_tmp_dir = os.path.join(tmp_dir, "java")
    runtime_dir = os.path.join(tmp_dir, "runtime")
    os.makedirs(mine_cache, exist_ok=True)
    os.makedirs(tmp_dir, exist_ok=True)
    os.makedirs(java_tmp_dir, exist_ok=True)
    os.makedirs(runtime_dir, mode=0o700, exist_ok=True)
    os.chmod(runtime_dir, 0o700)
    engine_link = os.path.join(mine_cache, "engine")
    if not os.path.islink(engine_link) and not os.path.exists(engine_link):
        os.symlink(config.SHARED_MINE_ENGINE, engine_link)

    cmd = [
        config.PYTHON, "-u", config.RUN_STREAMING_EVAL,
        "--model_path", config.BASE_MODEL,
        "--config", config_name,
        "--task_list", task_list,
        "--max_steps", str(max_steps),
        "--record_path", record_path,
        "--verbose",
    ]
    if lora_adapter_path:
        cmd += ["--lora_adapter_path", lora_adapter_path]
    env = _env()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["MINESTUDIO_DIR"] = mine_cache
    env["TMPDIR"] = tmp_dir
    env["XDG_RUNTIME_DIR"] = runtime_dir
    java_option = f"-Djava.io.tmpdir={java_tmp_dir}"
    env["JAVA_TOOL_OPTIONS"] = " ".join(
        x for x in (env.get("JAVA_TOOL_OPTIONS"), java_option) if x
    )
    if bench_seed is not None:
        env["OPENHA_BENCH_SEED"] = str(bench_seed)
    if extra_env:
        env.update(extra_env)
    log_file = log_file or os.path.join(config.LOGS_DIR, f"{config_name}.log")
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    with open(log_file, "a") as out:
        return subprocess.Popen(
            cmd,
            cwd=config.OPENHA_REPRO,
            env=env,
            stdout=out,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )


def build_data(output_dir, openha_success_root, icl_success_root, max_openha_rollouts=240,
               max_icl_rollouts=260, max_samples_per_rollout=12, tail_keep=16,
               allowed_tasks=None):
    """Build parser-constrained SFT data from successful rollouts.

    If `allowed_tasks` is provided, only success rollouts whose task is in that
    list are kept (used to enforce the frozen TRAIN_V2 / TEST_V2 split).
    """
    cmd = [
        config.PYTHON, "-u", config.BUILD_DATA_SCRIPT,
        "--output_dir", output_dir,
        "--openha_success_root", openha_success_root,
        "--icl_success_root", icl_success_root,
        "--max_openha_rollouts", str(max_openha_rollouts),
        "--max_icl_rollouts", str(max_icl_rollouts),
        "--max_samples_per_rollout", str(max_samples_per_rollout),
        "--tail_keep", str(tail_keep),
    ]
    if allowed_tasks:
        cmd += ["--allowed_tasks", allowed_tasks]
    env = _env()
    return subprocess.run(cmd, cwd=config.OPENHA_REPRO, env=env, check=False)


def train_lora(train_jsonl, eval_jsonl, output_dir, learning_rate=3e-6, lora_rank=8,
               lora_alpha=16, max_steps=150, gpu_id=0):
    """Train a LoRA adapter. Returns the subprocess handle (blocking)."""
    cmd = [
        config.PYTHON, "-u", config.TRAIN_SCRIPT,
        "--model_path", config.BASE_MODEL,
        "--train_jsonl", train_jsonl,
        "--eval_jsonl", eval_jsonl,
        "--output_dir", output_dir,
        "--learning_rate", str(learning_rate),
        "--lora_rank", str(lora_rank),
        "--lora_alpha", str(lora_alpha),
        "--gradient_accumulation_steps", "8",
        "--max_steps", str(max_steps),
        "--save_steps", "30",
        "--eval_steps", "30",
    ]
    env = _env()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    return subprocess.run(cmd, cwd=config.OPENHA_REPRO, env=env, check=False)


def find_free_gpu(threshold_mb=None):
    """Return the GPU index with the most free memory >= threshold, or None."""
    threshold_mb = threshold_mb or config.GPU_FREE_THRESHOLD_MB
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.used,memory.total", "--format=csv,noheader,nounits"],
            text=True,
        )
    except Exception:
        return None
    best = None
    best_free = -1
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:
            try:
                idx = int(parts[0])
            except ValueError:
                continue
            used, total = int(parts[1]), int(parts[2])
            free = total - used
            if free >= threshold_mb and free > best_free:
                best, best_free = idx, free
    return best
