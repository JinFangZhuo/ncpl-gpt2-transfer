# Portable NCPL-to-GPT-2 20-configuration campaign

This directory publishes the exact small source/configuration snapshot needed to
reproduce or continue the frozen GPT-2/OpenWebText campaign on another machine.
It intentionally excludes checkpoints, dataset caches, Python environments, and
live lock/state files.

The absolute `/jumbo/...` strings retained inside the two frozen JSON files are
provenance metadata only; the runtime controller does not dereference them. They
are intentionally unchanged so that the published SHA-256 freeze remains valid.

## Public dependencies

- FARMS: <https://github.com/HUST-AI-HYZ/FARMS>
- RMNP: <https://github.com/Dominator-Index/RMNP>
- NCPL source: <https://github.com/zhqwqwq/Configuration-to-Performance-Scaling-Law>
- NCPL weights: <https://huggingface.co/OptimizerStudy/NCPL-final>
- OpenWebText: <https://huggingface.co/datasets/Skylion007/openwebtext>

The bundled `ncpl_predictions.json` is frozen and already bound to the candidate
manifest. NCPL code/weights are therefore not required merely to train these 20
GPT-2 configurations; they are required only when independently reproducing the
NCPL prediction stage.

## Install

From the cloned FARMS repository:

```bash
cd GPT2_NCPL_Transfer
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# Install a CUDA/PyTorch build appropriate for the A100 host first.
python -m pip install -r requirements.txt
./setup_rmnp.sh
GPT2_CAMPAIGN_GPUS=0,1 python campaign/controller.py --check
```

`setup_rmnp.sh` clones RMNP at commit
`a662ebcfc5a76c23f6ddee7a0d9d4b883c847478` and overlays the exact modified
trainer/model files used by this campaign. Set `RMNP_REPO` if RMNP is installed
elsewhere. Set `GPT2_CACHE_ROOT` to move Hugging Face and TorchInductor caches to
fast local storage.

## Frozen inputs

```text
candidate_manifest.json  f674ebb0b3f0d848dc0da8a2d6a73f28f2600cb1710a511ab23e36dbc71bb9a4
ncpl_predictions.json    e87d7ef9c575b7dc6cd675360c7efc80faab1b13a65d7132eabba704bfdb98f7
```

Verify them before training:

```bash
cd campaign
sha256sum -c candidate_manifest.sha256
sha256sum -c ncpl_predictions.sha256
```

The committed `results_snapshot_10_completed.tsv` is a provenance snapshot, not
an automatically active result file. It contains the 10 completed L40 results
present when this package was created. Candidate index 10 was still running on
the original L40 server and is not included.

## Start four A100s

For a clean A100-only comparison, leave `campaign/results.tsv` absent and rerun
all 20 configurations. The launcher forms two 2-GPU DDP workers: GPUs 0-1 and
GPUs 2-3. Each run keeps world size 2, micro-batch 1, gradient accumulation 64,
global batch 128, sequence length 4096, seed 0, and 5,120 optimizer updates.

```bash
cd campaign
python launch_four_a100.py \
  --gpu-pairs '0,1;2,3' \
  --acknowledge-exclusive-campaign
```

The acknowledgement is mandatory because independent L40 and A100 controllers
cannot safely claim candidates from the same queue without a shared atomic lock.
Do not launch this controller while the original L40 campaign is still active.

To continue rather than rerun, first stop the original controller and copy its
latest `results.tsv` into this `campaign/` directory. Do not rely on the committed
snapshot if additional L40 runs have completed. Mixing L40 and A100 results is
allowed operationally but must be recorded as a hardware-domain change in the
scientific analysis; rerunning at least one completed anchor on A100 is strongly
recommended.

Create `campaign/STOP` for a controlled stop. Runtime outputs are ignored by Git.

The RMNP-derived overlay files are distributed under the upstream Apache-2.0
license reproduced in `RMNP_LICENSE`.
