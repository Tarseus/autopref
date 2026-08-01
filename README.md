# AutoPref

AutoPref searches training objectives for neural combinatorial optimization with a fixed two-stage workflow:

1. Search an executable preference loss while keeping pair construction fixed.
2. Freeze the selected loss and search pair weights on the fixed all-pairs construction.

## Requirements

- Python 3.10 or newer
- PyTorch with a CUDA build for GPU execution
- An OpenAI-compatible API endpoint

A GPU is recommended for the short-training fitness evaluations. Configuration validation can run on CPU.

## LLM configuration

Create `.env` in the repository root:

```dotenv
OPENAI_API_KEY=your-api-key
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=your-openai-model
```

## Run the search

```bash
python -m PTP.ptp_discovery.run_two_stage_search \
  --config xxx.yaml
```

Override the device from the command line when needed:

```bash
python -m PTP.ptp_discovery.run_two_stage_search \
  --config xxx.yaml \
  --device cuda:0
```

## Use the discovered objective

The final pair artifact can be loaded directly by POMO. The model resolves the matching loss and weighting artifacts from the same directory:

```python
from rl4co.envs import TSPEnv
from rl4co.models import POMO
from rl4co.utils import RL4COTrainer

pair_path = "runs/autopref_tsp20/stage2_weight/<run>/best_pair.json"

env = TSPEnv(generator_params={"num_loc": 20})
model = POMO(
    env,
    loss_type="free_loss",
    pref_pair_json_path=pair_path,
    num_augment=1,
    batch_size=64,
    train_data_size=100_000,
    val_data_size=10_000,
    test_data_size=10_000,
)
trainer = RL4COTrainer(max_epochs=100, accelerator="gpu", devices=1)
trainer.fit(model)
trainer.test(model)
```

Replace `<run>` with the Stage 2 directory recorded as `stage2_run` in `two_stage_manifest.json`.