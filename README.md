# microgpt
Playing with [karpathy's microgpt](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95)

## Quickstart

Train a model (creates a new run folder under `runs/`):

```/dev/null/command.txt#L1-1
python train.py
```

Run inference from a saved checkpoint:

```/dev/null/command.txt#L1-1
python infer.py --checkpoint runs/<run-name>/checkpoint.pkl
```

## Run Artifacts

Each training run writes:

- `runs/<run-name>/config.json` – exact config used
- `runs/<run-name>/metrics.csv` – loss curve
- `runs/<run-name>/checkpoint.pkl` – model + optimizer + metadata
- `runs/<run-name>/notes.txt` – scratchpad

## Configs

Create a config file to customize a run:

```/dev/null/config.json#L1-16
{
  "dataset_path": "input.txt",
  "seed": 42,
  "n_embd": 16,
  "n_head": 4,
  "n_layer": 1,
  "block_size": 16,
  "learning_rate": 0.01,
  "beta1": 0.85,
  "beta2": 0.99,
  "eps_adam": 1e-8,
  "num_steps": 1000,
  "log_every": 10,
  "temperature": 0.5,
  "num_samples": 100
}
```

Then run:

```/dev/null/command.txt#L1-1
python train.py --config path/to/config.json --run-name my_experiment
```

## Notes

- This refactor keeps the model “guts” explicit in pure Python.
- A Torch version can be bolted on later as a parallel implementation.
