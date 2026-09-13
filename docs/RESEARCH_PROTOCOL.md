# Cognitive transfer study: protocol and evidence status

## Status as of this revision

The training, evaluation, provenance, and comparison infrastructure is implemented. No completed multi-seed comparison is committed to this repository, so this document defines the intended study and does not report a transfer-learning result.

| Evidence component | Current state |
| --- | --- |
| Released task contracts and fixed task digests | Implemented |
| Development, validation, and final-test seed partitions | Implemented |
| Immutable run manifests and checkpoint hashes | Implemented |
| Scratch-versus-full-policy comparison validation | Implemented |
| Selective initialization methods | Implemented for experimentation |
| Complete multi-seed result table and uncertainty | Pending |
| Final-test analysis | Pending and held until the protocol is frozen |

## Research question

For a composite MiniGrid task, does initializing PPO from a policy trained on a simpler component task improve target-task learning efficiency or held-out success compared with target training from scratch under the same target budget?

The first target is `lava_door_key-v2`, whose mission combines key pickup, lava-gap navigation, door unlocking, and goal reaching. Candidate sources should isolate related skills, such as `key_pickup-v2`, `lava_navigation-v1`, and `door_key-v1`.

## Primary comparison

Each target seed should produce matched conditions that differ only in initialization:

1. **Scratch:** a newly initialized target policy.
2. **Full-policy transfer:** actor and critic parameters copied from one protected source checkpoint.

The governed comparison reader currently accepts these two conditions. Selective initializations exist in the training code, but they should be treated as later ablations until the same validation rules cover them end to end.

## Planned ablations

After the primary comparison is complete, evaluate which transferred components matter:

- actor-body fine-tuning;
- a frozen-actor probe;
- full actor with a fresh critic; and
- actor body with a fresh action head and copied critic.

These variants should use the same source checkpoint, target seeds, target budget, evaluation grid, and environment contract as the matched scratch condition.

## Fixed partitions and budgets

The values below are defined in `training/benchmark_contracts.py`:

- Development seeds: 10 seeds, `11001` through `11010`.
- Validation seeds: 40 seeds, `12001` through `12040`.
- Final-test seeds: 40 seeds, `13001` through `13040`.
- Source training budget: 1,024,000 timesteps.
- `lava_door_key-v2` target budget: 2,048,000 timesteps.
- Pilot ceiling per target run: 3,072,000 timesteps.
- Periodic evaluation overhead ceiling: 15% of elapsed runtime.

Development seeds support iteration and learning curves. Validation seeds support model comparison. Final-test seeds remain untouched until conditions and analysis choices are frozen.

## Outcomes

### Primary

- normalized area under the Development success curve;
- sustained time to 50% and 80% success, defined as the second point in the first two consecutive points at or above the threshold; and
- success count and rate on the held-out Validation seeds.

### Secondary

- final Development success rate;
- mean reward and episode length at each fixed evaluation point;
- runtime and evaluation-overhead fraction; and
- failure or evaluation-error counts.

## Replication and uncertainty

A result table should include every planned training seed, not only the strongest run. Aggregate reporting should include the number of completed seeds, mean or median effect, dispersion, and the individual seed-level effects. Confidence intervals should be produced by a declared resampling method once the replicate count is fixed.

Any run that violates the task contract, seed partition, target budget, artifact hashes, or complete evaluation grid must be excluded with a recorded reason. The comparison loader already fails closed on several of these conditions.

## Result table to publish

| Source task | Initialization | Replicates | Dev AUC | Sustained T50 | Sustained T80 | Validation success | Effect vs scratch |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Pending | Scratch | — | — | — | — | — | reference |
| Pending | Full-policy transfer | — | — | — | — | — | — |

The table remains intentionally empty until complete evidence bundles exist.

## Claim boundary

The repository can currently support claims about software behavior: experiment configuration is recorded, artifacts are content-addressed, evaluation identities are separated, and comparison bundles are validated. It cannot yet support a claim that transfer improves or harms learning. That conclusion requires the completed replicated study described above.

