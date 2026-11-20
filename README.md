# CoMA

Official PyTorch implementation of **[Concept-Enhanced Multi-Scale Cross-Modal Alignment for Medical Visual Representation Learning](https://ieeexplore.ieee.org/document/11705661)** (IEEE JBHI, Early Access).

> Xiangmin Kong, Xibin Jia, Dawei Yang, Zhenghan Yang

<p align="center">
  <img src="assets/framework.png" width="100%" alt="CoMA framework overview"/>
</p>

CoMA is a medical vision–language pre-training (MedVLP) framework that extracts **concept clauses** from radiology reports and aligns them with images at multiple granularities:

- **Concept Clause Decomposition (CCD)** — decompose free-text reports into semantically complete medical concept clauses
- **Global Instance-level Alignment (GIA)** — image–report contrastive alignment
- **Concept-level Token-wise Alignment (CTA)** — bidirectional cross-attention between image patches and concept clauses
- **Bi-level Prototype Alignment (IPA / CPA)** — cross-modal prototype clustering at instance and concept levels

## Requirements

```bash
conda create -n coma python=3.8 -y
conda activate coma
# Install a CUDA-compatible torch / torchvision build first, then:
pip install -r requirements.txt
pip install -e .
```

Place [Bio_ClinicalBERT](https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT) under `model_files/Bio_ClinicalBERT/` (or use a local / Hugging Face cache path).

## Datasets

| Dataset | Usage | Download |
| --- | --- | --- |
| [MIMIC-CXR-JPG](https://physionet.org/content/mimic-cxr-jpg/2.0.0/) | Pre-training | PhysioNet (credentialed access) |
| [CheXpert](https://stanfordmlgroup.github.io/competitions/chexpert/) | Classification / zero-shot | Stanford ML Group |
| [RSNA Pneumonia](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge) | Classification / segmentation / detection | Kaggle |
| [SIIM-ACR Pneumothorax](https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation) | Segmentation | Kaggle |
| [COVIDx](https://github.com/lindawangg/COVID-Net) | Classification / zero-shot | COVID-Net |
| [Object-CXR](https://jfhealthcare.github.io/object-CXR/) | Detection | JF Healthcare |

Set the dataset root, then preprocess:

```bash
export COMA_DATA_DIR=/path/to/dataset
# optional: export COMA_OUTPUT_DIR=/path/to/outputs
bash preprocess.sh
```

Concept Clause Decomposition for MIMIC-CXR reports:

```bash
python ./coma/preprocess/mimic_cxr_rewritereport.py
```

## Pre-training

```bash
bash run_pretrain.sh
```

## Downstream Evaluation

Set `CKPT=/path/to/coma.ckpt`, then run:

```bash
# Classification (CheXpert / RSNA / COVIDx)
CKPT=/path/to/coma.ckpt bash run_downstream_classifier.sh

# Segmentation (RSNA / SIIM)
CKPT=/path/to/coma.ckpt bash run_downstream_segmenter.sh

# Detection (RSNA / Object-CXR)
CKPT=/path/to/coma.ckpt bash run_downstream_detector.sh
```

Zero-shot classification:

```bash
python ./coma/models/downstream/zero-shot.py --model coma --dataset chexpert --ckpt_path /path/to/coma.ckpt
```

## Citation

If you use this code, please cite:

```bibtex
@article{kong2026coma,
  title   = {Concept-Enhanced Multi-Scale Cross-Modal Alignment for Medical Visual Representation Learning},
  author  = {Kong, Xiangmin and Jia, Xibin and Yang, Dawei and Yang, Zhenghan},
  journal = {IEEE Journal of Biomedical and Health Informatics},
  year    = {2026},
  note    = {Early Access},
  url     = {https://ieeexplore.ieee.org/document/11705661}
}
```

## Acknowledgements

This implementation builds on the medical VLP community (e.g., ConVIRT, GLoRIA, MGCA, PRIOR, MedKLIP, KAD). Please obtain datasets from their original providers under the corresponding licenses and data-use agreements.

## License

This repository is released under the [MIT License](LICENSE).
