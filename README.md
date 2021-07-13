# Real-time Domain Adaptation in Semantic Segmentation

Andrea Cognolato*, Clauda Cuttano*, Cristina Tortia*

\**All authors have contributed equally*

## Notebooks

- `1-BiSeNet-training`: notebook to run train BiSeNet. Used for comparing epochs and backbones
- `2-IDDA-Loader`: notebook to explore and understand the IDDA dataset. Used for writing the actual dataloader
- `3-adversarial-training`: notebook to train and evaluate BiSeNet with output space adversarial domain adaptation
- `4-FDA-training`: notebook to train and evaluate BiSeNet with FDA. Its results do not appear in the final report
- `5-FDA-adversarial-training`: notebook to train and evaluate BiSeNet with adversarial domain adaptation and FDA
- `6-FDA-MBT+PSU`: notebbok to load 3 FDA+adversarially-trained BiSeNet models and compute their average predictions. Using these predictions we apply a thresolding to generate the pseudolabels needed for self-supervised training.
- `7-FDA-adversarial-training-PSU`: notebook to train and evaluate BiSeNet with adversarial domain adaptation and self-supervised FDA
