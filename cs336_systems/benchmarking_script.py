from benchmark import (
  MODELS,
  ModelConfig,
  compute_mean_and_std_pass_times,
  instantiate_model,
  model_size_mb,
)

warmup_steps = 5
model_name = "large" 

def compute_benchmarks(model_config: ModelConfig, warmup_steps: int):
  return {
    "without_sync_without_backward": compute_mean_and_std_pass_times(
      model_config = model_config,
      context_length = 256,
      measure_also_backward = False,
      synchronize = False,
      warmup_steps = warmup_steps,
    ),
    "with_sync_without_backward": compute_mean_and_std_pass_times(
      model_config = model_config,
      context_length = 256,
      measure_also_backward = False,
      synchronize = True,
      warmup_steps = warmup_steps,
    ),
    "with_sync_with_backward": compute_mean_and_std_pass_times(
      model_config = model_config,
      context_length = 256,
      measure_also_backward = True,
      synchronize = True,
      warmup_steps = warmup_steps,
    )
  }

def print_benchmarks(benchmarks: dict):
  for name, res in benchmarks.items():
    mean, std = res
    print(f'{name}: {mean:.4f}s \u00b1 {std:.4f}s')


config = MODELS[model_name]
total_size_mb = model_size_mb(instantiate_model(config, 256))
print(f"--- {model_name} model ({total_size_mb:.2f} MB) ---")
benchmarks= compute_benchmarks(config, warmup_steps = warmup_steps)
print_benchmarks(benchmarks)
