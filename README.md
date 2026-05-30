# Is Your LLM Overcharging You? Tokenization, Transparency, and Incentives

This repository contains the code for the paper ["Is Your LLM Overcharging You? Tokenization, Transparency, and Incentives"](https://arxiv.org/abs/2505.21627)
by Ander Artola Velasco, Stratis Tsirtsis, Nastaran Okati and Manuel Gomez-Rodriguez.


## Paper abstract

State-of-the-art large language models require specialized hardware and substantial energy to operate. As a consequence, cloud-based services that provide access to large language models have become very popular. In these services, the price users pay for an output provided by a model depends on the number of tokens the model uses to generate it—they pay a fixed price per token. In this work, we show that this pricing mechanism creates a financial incentive for providers to strategize and misreport the (number of) tokens a model used to generate an output, and users cannot prove, or even know, whether a provider is overcharging them. However, we also show that, if an unfaithful provider is obliged to be transparent about the generative process used by the model, misreporting optimally without raising suspicion is hard. Nevertheless, as a proof-of-concept, we introduce an efficient heuristic algorithm that allows providers to significantly overcharge users without raising suspicion, highlighting the vulnerability of users under the current pay-per-token pricing mechanism. Further, to completely eliminate the financial incentive to strategize, we introduce a simple incentive-compatible token pricing mechanism. Under this mechanism, the price users pay for an output provided by a model depends on the number of characters of the output—they pay a fixed price per character. Along the way, to illustrate and complement our theoretical results, we conduct experiments with several large language models from the ``Llama``, ``Gemma`` and ``Mistral`` families, and input prompts from the LMSYS Chatbot Arena platform.


## Dependencies

All experiments were performed using Python 3.11.2. To create a virtual environment and install the project dependencies:

```bash
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```


## Repository structure

```
├── data
│   └── LMSYS.txt
├── figures
│   ├── fixed_string
│   └── heur
├── notebooks
│   ├── appendix_example.ipynb
│   ├── cpt.ipynb
│   ├── energy_profit_plots.ipynb
│   ├── plot_profit_no_transparency.ipynb
│   ├── plots_heur_new.ipynb
│   ├── plots_heur_random.ipynb
│   └── process_ds.ipynb
├── outputs
│   ├── cpt              (JSON results from LMSYS_generation.py)
│   ├── energy_outputs   (JSON results from energy.py)
│   └── heuristic_new    (PKL results from heuristic_misreporting.py — regenerate locally)
├── scripts
│   ├── pkl_to_json.py
│   ├── script_slurm_heur.sh
│   ├── script_slurm_lmsys.sh
│   ├── script_slurm_lmsys_energy.sh
│   ├── script_slurm_lmsys_generation.sh
│   └── script_slurm_lmsys_generation_loop.sh
└── src
    ├── energy.py
    ├── heuristic_misreporting.py
    ├── LMSYS_generation.py
    ├── tokenizations.py
    ├── tokenizations_fixed.py
    ├── tokenizations_fixed_plausible.py
    └── utils.py
```

- `data` contains the LMSYS prompt set used in the experiments.
- `figures` contains all the figures presented in the paper.
- `notebooks` contains the Jupyter notebooks that generate the figures:
  - [cpt.ipynb](notebooks/cpt.ipynb) — character-per-token price and provider-margin distributions across models, temperatures, top-p values, and languages.
  - [plots_heur_new.ipynb](notebooks/plots_heur_new.ipynb) — per-model "plausible sequences" and "overcharged tokens" curves from the heuristic-misreporting evaluation.
  - [plots_heur_random.ipynb](notebooks/plots_heur_random.ipynb) — random-variant heuristic plausibility curve.
  - [plot_profit_no_transparency.ipynb](notebooks/plot_profit_no_transparency.ipynb) — random-split profit increase as a function of split iterations.
  - [energy_profit_plots.ipynb](notebooks/energy_profit_plots.ipynb) — per-call energy demo, per-model energy statistics, and the profit / utility plots that combine energy ratios with heuristic-misreporting results.
  - [process_ds.ipynb](notebooks/process_ds.ipynb) — builds the LMSYS prompt set used by the other notebooks.
  - [appendix_example.ipynb](notebooks/appendix_example.ipynb) — generates the worked examples in Appendix C.2 of the paper.
- `outputs` contains intermediate files produced by the experiment scripts and consumed by the notebooks:
  - `cpt` — generation outputs from `LMSYS_generation.py`, shipped as `.json`.
  - `energy_outputs` — GPU energy results from `energy.py`, shipped as `.json`.
  - `heuristic_new` — plausibility counts from `heuristic_misreporting.py`. These are `.pkl` files and are excluded from the repository (see *Reproducing intermediate outputs* below).
- `scripts` contains SLURM submission templates and a small utility:
  - [script_slurm_heur.sh](scripts/script_slurm_heur.sh) — runs `heuristic_misreporting.py` on the LMSYS prompts.
  - [script_slurm_lmsys_generation.sh](scripts/script_slurm_lmsys_generation.sh) — runs `LMSYS_generation.py` on the LMSYS prompts.
  - [script_slurm_lmsys_generation_loop.sh](scripts/script_slurm_lmsys_generation_loop.sh) — loops `LMSYS_generation.py` over a grid of (language, temperature, top-p, model).
  - [script_slurm_lmsys_energy.sh](scripts/script_slurm_lmsys_energy.sh) — runs `energy.py` for GPU energy measurement.
  - [script_slurm_lmsys.sh](scripts/script_slurm_lmsys.sh) — convenience submission template kept for reproducibility.
  - [pkl_to_json.py](scripts/pkl_to_json.py) — converts a directory of `.pkl` outputs into portable `.json` files (used to produce the JSON snapshots shipped under `outputs/cpt/` and `outputs/energy_outputs/`).
- `src` contains the source code used to produce the experiment outputs:
  - [LMSYS_generation.py](src/LMSYS_generation.py) — generates outputs from a HuggingFace model under top-p or top-k sampling for a list of prompts.
  - [heuristic_misreporting.py](src/heuristic_misreporting.py) — main script of the heuristic experiments. It generates outputs, iteratively splits tokens, and checks whether each depth still satisfies the original top-p / top-k sampling condition.
  - [energy.py](src/energy.py) — measures GPU energy during one generation call and one scoring forward pass.
  - [tokenizations_fixed.py](src/tokenizations_fixed.py) — enumerates all tokenizations of a fixed output string and computes the model's conditional probability of each.
  - [tokenizations_fixed_plausible.py](src/tokenizations_fixed_plausible.py) — same enumeration but flags which tokenizations are top-p / top-k plausible.
  - [tokenizations.py](src/tokenizations.py) — helper routines used by the above (enumerate tokenizations, compute tokenization probability, verify sampling conditions).
  - [utils.py](src/utils.py) — plotting helpers (LaTeX-friendly figure dimensions) and the optimal-tokenization (shortest-cover) routine.


### Reproducing intermediate outputs

`.pkl` files are excluded from the repository via [.gitignore](.gitignore).
JSON snapshots of the `outputs/cpt/` and `outputs/energy_outputs/` runs
are included so that the corresponding notebooks can be re-executed from
a fresh clone. The heuristic-misreporting results in
`outputs/heuristic_new/` are regenerated by running
`heuristic_misreporting.py` (or the SLURM submission templates above).
`pkl_to_json.py` can be used to mirror new pickle results to JSON.


## Instructions

### Downloading the models

Our experiments use LLMs from the Llama, Gemma and Mistral families. These are gated models and require accepting the corresponding license. Request access at:
- [meta-llama/Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)
- [google/gemma-3-4b-it](https://huggingface.co/google/gemma-3-4b-it)
- [mistralai/Ministral-8B-Instruct-2410](https://huggingface.co/mistralai/Ministral-8B-Instruct-2410)

Once you have access, authenticate with `huggingface-cli login` and the scripts will cache the model weights under `models/` (which is excluded from the repository via [.gitignore](.gitignore)).


### Fixed-string experiment

The script [tokenizations_fixed_plausible.py](src/tokenizations_fixed_plausible.py) produces the data behind Figure 1: for a given output string and prompt, it returns the number of top-p / top-k plausible tokenizations.


### LMSYS heuristic-misreporting experiment

The main heuristic-misreporting experiment runs [heuristic_misreporting.py](src/heuristic_misreporting.py). It can be invoked locally or via the cluster template [script_slurm_heur.sh](scripts/script_slurm_heur.sh) (which reads prompts from [data/LMSYS.txt](data/LMSYS.txt)). Key flags:
- `--model` sets the HuggingFace model id (e.g. `meta-llama/Llama-3.2-1B-Instruct`).
- `--temperature` sets the sampling temperature.
- `--p` / `--k` set top-p or top-k.
- `--prompts` is the list of prompt strings to process.
- `--splits` is the list of iteration depths at which to record plausibility.

To produce the per-model heuristic-misreporting figures from the resulting pickles, run [plots_heur_new.ipynb](notebooks/plots_heur_new.ipynb).


### Energy experiment

[energy.py](src/energy.py) measures the GPU energy of one `generate()` call and one scoring forward pass for a list of prompts, sampling NVML power draw at 10 Hz. The aggregated per-model results (shipped under `outputs/energy_outputs/`) are consumed by [energy_profit_plots.ipynb](notebooks/energy_profit_plots.ipynb).


### CPT (character-per-token) experiment

[LMSYS_generation.py](src/LMSYS_generation.py) generates outputs under a chosen sampling configuration. The resulting outputs (shipped under `outputs/cpt/`) are loaded by [cpt.ipynb](notebooks/cpt.ipynb) to compute character-per-token price and provider-margin distributions across models, temperatures, top-p values, and languages. [script_slurm_lmsys_generation_loop.sh](scripts/script_slurm_lmsys_generation_loop.sh) sweeps the full (language, temperature, top-p, model) grid in one submission.


## Contact & attribution

In case you have questions about the code, you identify potential bugs or you would like us to include additional functionalities, feel free to open an issue or contact [Ander Artola Velasco](mailto:avelasco@mpi-sws.org).

If you use parts of the code in this repository for your own research, please consider citing:

```bibtex
@misc{velasco2026llmoverchargingyoutokenization,
      title={Is Your LLM Overcharging You? Tokenization, Transparency, and Incentives},
      author={Ander Artola Velasco and Stratis Tsirtsis and Nastaran Okati and Manuel Gomez-Rodriguez},
      year={2026},
      eprint={2505.21627},
      archivePrefix={arXiv},
      primaryClass={cs.GT},
      url={https://arxiv.org/abs/2505.21627},
}
```
