# IEEE Grid Data Generator

`ieee-grid-data-generator` is a lightweight toolkit for generating power-system datasets from IEEE test cases. It focuses on producing **clean, consistent raw power-flow samples** and converting them into **model-agnostic records** that downstream ML pipelines can transform into their preferred formats.

---

## What this repo does

- Generates n-k contingency samples from IEEE cases using PyPower AC power flow.
- Organizes results into a predictable on-disk layout (`results.pkl` per topology/level).
- Converts raw samples into a list-of-records dataset via pluggable processors.

What it does **not** do:
- Train models or enforce a graph framework.
- Perform model-side batching/normalization.

---

## Repository layout

```
.
├── generate_n_k_data.py      # Generate raw PF samples (n-k contingencies)
├── build_dataset_from_raw.py # Convert raw samples into dataset records
├── raw_reader.py             # Read raw results.pkl groups
├── processors.py             # Feature processors (x/y/branch_attr/matrix_attr)
└── overview.py               # Topology enumeration + connectivity filtering
```

---

## Requirements

- Python 3.9+
- numpy
- pypower
- tqdm

```bash
pip install numpy pypower tqdm
```

---

## 1) Generate raw data

This step enumerates (or samples) **connected** topologies, perturbs loads/generators, runs AC PF, and stores results grouped by topology and power level.

```bash
python generate_n_k_data.py \
  --case IEEE39 \
  --k 1 \
  --max_size 100 \
  --power_level 0.8 0.9 1.0 1.1 1.2 \
  --samples_per_level 200 200 200 200 200 \
  --raw_data_dir ./data/raw
```

Useful flags:
- `--keep_failed`: keep failed PF samples instead of dropping them.
- `--sample_max_tries`: cap sampling attempts when the topology space is truncated.
- `--preview_topologies`: estimate connected ratio for reporting.

### Output layout

```
./data/raw/IEEE39/k=1/
  topo_000000/
    level_0.800/results.pkl
    level_0.900/results.pkl
    ...
  topo_000001/
    level_0.800/results.pkl
    ...
```

Each `results.pkl` contains:
- `meta`: case, k, topology id, power level, outage branches, etc.
- `samples`: list of `{sample_id, success, results}`

---

## 2) Build a dataset

Convert raw groups into a list of records using processors defined in `processors.py`.

```bash
python build_dataset_from_raw.py \
  --raw_root ./data/raw \
  --case IEEE39 \
  --k 1 \
  --processors meta raw_results x y branch_attr matrix_attr \
  --save_path ./data/ieee39_k1.pkl
```

Optional filters:
- `--levels 0.9 1.0`
- `--max_groups 10`
- `--max_samples_per_group 50`

---

## Dataset format

The dataset is saved as a Python list of records:

```python
Dataset = List[Dict[str, Any]]
```

Records are assembled by processors, for example:
- `meta`: compact metadata per sample
- `raw_results`: raw PyPower results dict
- `x`: node features `[p, q, v, θ, bus_type]`
- `y`: targets `[v, θ, p, q]`
- `branch_attr`: sparse incidence + branch attributes
- `matrix_attr`: sparse incidence + electrical matrix attributes

Select processors with `--processors` when running `build_dataset_from_raw.py`.

---

## Custom processors

Add your own processor in `processors.py`:

```python
from processors import register_processor

@register_processor("my_feature")
def proc_my_feature(record, group_meta, sample, pkl_path):
    record["my_feature"] = ...
```

Then use:

```bash
python build_dataset_from_raw.py --processors meta my_feature
```

---

## Topology generation notes

Topology enumeration and connectivity filtering live in `overview.py`:
- k-branch outage sets are enumerated or sampled.
- Disconnected topologies are dropped.
- The generator records outage branch indices per topology.

If the combinatorial space is large, sampling is used up to `--max_size`.

---

