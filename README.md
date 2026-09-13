# TransferGrid

TransferGrid is a local Streamlit workbench for training, evaluating, transferring and replaying PPO agents in MiniGrid.

![TransferGrid Compact Workbench](docs/assets/workbench.png)

## What it does

- **Train** a PPO policy from scratch on one of the included grid-world tasks.
- **Evaluate** a saved checkpoint with deterministic validation episodes.
- **Transfer** a full policy or selectively reset the action head and critic.
- **Replay** recorded episodes with play, pause, step, reset and speed controls.

Long-running jobs execute outside the Streamlit render loop. Checkpoints carry configuration and provenance metadata, while evaluation reports remain linked to the exact checkpoint they inspect.

## Run locally

Requirements: Python 3.11+ and Windows, Linux or macOS.

```bash
git clone https://github.com/Szaman07/CSE299_TransferGrid.git
cd CSE299_TransferGrid
python -m venv .venv
```

Activate the environment, then install and launch:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The repository starts without bundled experiment data. Train a checkpoint locally to populate the artifact-driven workflows.

## Architecture

![TransferGrid architecture](docs/assets/architecture.png)

| Layer | Responsibility |
|---|---|
| `ui/` | Compact Streamlit workflow and interaction state |
| `environments/` | Task catalog, construction and rendering |
| `training/` | PPO execution, transfer initialization, evaluation and replay |
| `models/`, `metrics/` | Model and measurement utilities |

Generated checkpoints, reports, logs and local experiment data are intentionally ignored so a clone stays lightweight.

## Research status

TransferGrid currently provides the experiment infrastructure for a cognitive-transfer study; the repository does **not** yet publish a completed multi-seed result table. The code already fixes task identities, seed partitions, training budgets, checkpoint provenance, evaluation records, and strict comparison bundles so later claims can be traced to immutable artifacts.

The planned first comparison asks whether a PPO policy transferred from a simpler source task learns the composite `lava_door_key-v2` target more efficiently than a matched scratch policy. The primary evidence will be learning-curve area, sustained success thresholds, and held-out validation success—not a single best run.

Read the [research protocol and claim boundary](docs/RESEARCH_PROTOCOL.md) for the proposed baselines, seeds, metrics, ablations, and reporting rules. Results will be added only after the full protocol has been run; empty templates or selected screenshots will not be presented as findings.

## Technical evidence

- `training/benchmark_contracts.py` freezes development, validation, and final-test seed partitions plus compute budgets.
- `training/experiments.py` writes immutable run-centered configuration, summary, manifest, and artifact tables.
- `training/comparison.py` rejects incomplete or scientifically mixed bundles and recomputes reported effects.
- `training/provenance.py` and `training/data_integrity.py` bind evidence to source revisions and content hashes.
- `environments/task_contracts/` records task-level observation, action, and mission contracts.

The practical contribution is the workbench and evidence pipeline. Any future scientific conclusion will be reported separately with its replicate count and uncertainty.

## License

[MIT](LICENSE)

