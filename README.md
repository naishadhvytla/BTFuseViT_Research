# BTFuseViT: A Vision Transformer with Multi-Dataset Fusion and Phased Transfer Learning for Multi-Class Brain Tumor Classification

Code and released results for the BTFuseViT paper (manuscript under review, *BMC Medical Imaging*).

BTFuseViT is a single ViT-Base/16 (384×384 input, 86.6 M parameters) that classifies brain MRI slices into glioma, meningioma, pituitary tumor and no tumor. It is trained on the Nickparvar dataset, TCGA-LGG FLAIR crops and BRISC 2025, with White Stripe intensity normalization. Fine-tuning runs in three phases: head warm-up, a 4-tier discriminative learning rate, and stochastic weight averaging (SWA). Predictions use temperature scaling and 14-view test-time augmentation (TTA). The evaluation protocol controls for leakage. Every pair of images across all sources is screened for duplicates. Any training image linked to a test image is removed. A grouped validation split is used for every model-selection and calibration decision, so the test sets are used only once.

## Key results

**Leakage-controlled protocol** (main results of the paper)

| Test set | Images | Accuracy (95% CI) | Macro-F1 | Macro-AUC |
|---|---:|---|---:|---:|
| Nickparvar test | 1,600 | 95.19% (94.03–96.13) | 95.18% | 0.9903 |
| BRISC 2025 test | 1,000 | 97.50% (96.34–98.30) | 97.42% | — |
| PMRAM (external) | 1,505 | 96.48% (95.42–97.30) | — | — |
| PMRAM images without a public copy | 524 | 95.80% | — | — |

**Standard image-level protocol.** This uses the same model with no duplicate removal and is included only for comparison with published work.

| Test set | Accuracy |
|---|---|
| Nickparvar test | 96.12% (95.06–96.97) |
| BRISC 2025 test | 99.20% |
| PMRAM | 98.87% |

**Duplicate audit.** The public data contain a copy of, or a closely related image to, 73.7% of the Nickparvar test images, 65.3% of the BRISC test images and 65.2% of the PMRAM images.

All numbers in the paper are in `results/generated/results_macros.tex` and `results/results/results.json`.

## Repository structure

```
BTFuseViT_Research/
├── README.md
├── LICENSE
├── CITATION.cff                         # citation metadata
├── requirements.txt                     # package versions used for the paper
├── notebooks/
│   ├── BTFuseViT_v6.ipynb               # final notebook that produced the paper (executed, outputs kept)
│   └── archive/
│       ├── BTFuseViT_v6_first_full_run.ipynb   # earlier full run, before the standard-protocol run was added
│       └── BTFuseViT_v6_smoke_test.ipynb       # executed smoke test (SMOKE_TEST = True)
└── results/                             # unpacked BTFuseViT_v6_outputs.zip of the final run
    ├── generated/                       # LaTeX tables (tab_*.tex) and all numeric macros (results_macros.tex)
    ├── figures/                         # paper figures (PDF + PNG)
    ├── manifests/
    │   └── partition_manifest.csv       # source, file, patient ID and role of every image
    ├── results/
    │   ├── near_duplicate_pairs.csv     # all duplicate / near-duplicate image pairs found by the screen
    │   ├── overlap_summary.csv          # overlap counts between every pair of sources
    │   ├── label_conflicts.csv          # duplicate pairs that carry different class labels
    │   ├── figshare_pid_map.csv         # figshare patient IDs recovered for Nickparvar / BRISC images
    │   ├── pmram_public_matches.csv     # PMRAM images with / without a copy in the public data
    │   ├── gradcam_localisation.csv     # per-image Grad-CAM++ localisation against BRISC tumor masks
    │   ├── preprocessing.json           # White Stripe windows and fallback rates
    │   ├── compute.json                 # parameters, FLOPs, latency and memory
    │   ├── environment.json             # package versions of the final session
    │   └── results.json                 # every reported number, configuration and run plan
    └── runs/                            # one folder per training (main, ablations, patient-disjoint, std_protocol)
        └── <run>/
            ├── summary.json             # configuration, selected checkpoint, temperature, timing, versions
            ├── history.json             # per-epoch training history
            ├── val_logits.npz           # validation logits used for selection and calibration
            └── logits_*.npz             # per-image test logits (14 TTA views) for each test set
```

The nine runs in `results/runs/` are:

| Folder | Configuration |
|---|---|
| `main` | Full model (BTFuseViT) |
| `main_patient_disjoint` | Patient-disjoint training (figshare patient IDs) |
| `abl_no_lgg` | Without LGG crops |
| `abl_no_brisc` | Without BRISC-train |
| `abl_nick_only` | Nickparvar-train only |
| `abl_no_boost` | Without glioma boost (class-balanced only) |
| `abl_uniform` | Without weighted sampling (uniform) |
| `abl_percentile` | Percentile normalization instead of White Stripe |
| `std_protocol` | Standard image-level protocol (no duplicate removal) |

## Which notebook to use

**`notebooks/BTFuseViT_v6.ipynb`** is the notebook that produced the paper. It covers everything end to end: download, duplicate screen, partitioning, preprocessing, all nine trainings, the analyses, the figures and the LaTeX tables.

The notebooks in `notebooks/archive/` are earlier executed records, kept for transparency. One is the first full run, made before the standard-protocol comparison was added; the other is the smoke test. All notebooks keep their cell outputs as the executed record.

## How to reproduce

1. Open `notebooks/BTFuseViT_v6.ipynb` in Google Colab.
2. Choose **Runtime → Change runtime type → GPU**. The paper used an NVIDIA RTX PRO 6000 Blackwell ("G4"); an A100 also works.
3. Choose **Runtime → Run all**. No edits are needed.

The notebook mounts Google Drive and saves results to `MyDrive/BTFuseViT_v6`. If Colab disconnects, reconnect and **Run all** again; finished trainings are reused rather than retrained.

A full run takes about 3–4 h on the RTX PRO 6000: the 9 trainings take about 1.7 GPU-hours, and duplicate screening, preprocessing and the analyses take the rest. To check that the code runs on a small subset in about 10 minutes, set `SMOKE_TEST = True` in the configuration cell.

The run writes `MyDrive/BTFuseViT_v6/BTFuseViT_v6_outputs.zip`, which holds every table (LaTeX), figure and number in the paper. The `results/` folder of this repository is that file, unpacked, from the final run.

## Datasets

The notebook downloads all datasets automatically with [kagglehub](https://github.com/Kaggle/kagglehub). They are **not redistributed** in this repository. Users must follow each dataset's license and terms of use.

| Dataset | Kaggle handle | Use |
|---|---|---|
| Nickparvar Brain Tumor MRI Dataset (version 2) | `masoudnickparvar/brain-tumor-mri-dataset` | training + test |
| LGG MRI Segmentation (TCGA-LGG) | `mateuszbuda/lgg-mri-segmentation` | training (FLAIR crops) |
| BRISC 2025 | `briscdataset/brisc2025` | training + test |
| figshare brain tumor dataset (Cheng et al., 2015) | `ashkhagan/figshare-brain-tumor-dataset` | only to recover patient identifiers |
| PMRAM Bangladeshi Brain Cancer MRI Dataset | `orvile/pmram-bangladeshi-brain-cancer-mri-dataset` | external test only |

- **LGG / TCGA-LGG:** if you use this dataset, cite Buda et al. (2019) and the TCGA-LGG collection on The Cancer Imaging Archive (TCIA).
- **PMRAM:** also available on Mendeley Data, doi:[10.17632/m7w55sw88b.1](https://doi.org/10.17632/m7w55sw88b.1).
- **figshare (Cheng et al., 2015):** used only to link Nickparvar / BRISC images to patient identifiers; it is not used for training or testing.

## Released results

Everything in `results/` was produced by the final notebook run:

- `manifests/partition_manifest.csv`: the source, file, patient ID and role of every image: training, validation, test, shortcut probe, or excluded (duplicate of a test image, duplicate within the training pool, or label conflict).
- `results/near_duplicate_pairs.csv` and `results/overlap_summary.csv`: the full output of the all-pairs duplicate screen and the overlap counts derived from it.
- `runs/*/logits_*.npz`: per-image test predictions (logits for all 14 TTA views, labels and image IDs) for every run and test set. Any metric in the paper can be recomputed from these files.
- `runs/*/summary.json` and `runs/*/history.json`: run summaries and training histories.
- `generated/`: the generated LaTeX tables and `results_macros.tex`.
- `figures/`: the paper figures. A few of them show example MRI slices from the public datasets as illustrations.

No model weights (`*.pt` / `*.pth`) are included.

## Software versions

Python 3.13, PyTorch 2.11.0, timm 1.0.29 (main runs; 1.0.30 for the standard-protocol run), grad-cam 1.5.7, scikit-learn 1.6.1, SciPy 1.16.3, statsmodels 0.15.0, NumPy 2.1.3, pandas 2.2.3, kagglehub 1.0.2.

The exact versions for each run are in `results/runs/*/summary.json` and `results/results/environment.json`. Pinned versions are listed in `requirements.txt`. On Colab, PyTorch and torchvision are preinstalled, and the notebook installs the remaining packages itself.

## Citation

> Kiran Kumar Eepuri, Naishadh Vytla, K. Rajendra Prasad, Pardhu Thottempudi, Kiran Siripuri. *BTFuseViT: A Vision Transformer with Multi-Dataset Fusion and Phased Transfer Learning for Multi-Class Brain Tumor Classification.* Manuscript under review, BMC Medical Imaging (2026).

No DOI is available yet. Machine-readable metadata is in [`CITATION.cff`](CITATION.cff); GitHub's "Cite this repository" button uses it.

## License

The code in this repository is released under the [MIT License](LICENSE). The datasets are not covered by this license; they remain under their own terms (see [Datasets](#datasets)).

## Disclaimer

This is research code for benchmark evaluation only. It is **not a medical device** and has **not been validated for clinical use**. Do not use it for diagnosis or treatment decisions.
