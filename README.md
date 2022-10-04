# What You See is What You Get: Distributional Generalization for Algorithm Design in Deep Learning

This repository contains the code of the experiments in the paper

[What You See is What You Get: Distributional Generalization for Algorithm Design in Deep Learning](https://arxiv.org/abs/2204.03230)

Authors:
[Bogdan Kulynych](https://bogdankulynych.me/),
[Yao-Yuan Yang](http://yyyang.me/),
[Yaodong Yu](https://yaodongyu.github.io/),
[Jarosław Błasiok](https://scholar.harvard.edu/jblasiok/home),
[Preetum Nakkiran](https://preetum.nakkiran.org/)

To appear in NeurIPS 2022

## Abstract

We investigate and leverage a connection between Differential Privacy (DP) and the recently proposed notion of Distributional Generalization (DG).
Applying this connection, we introduce new conceptual tools for designing deep-learning methods that bypass "pathologies" of standard stochastic gradient descent (SGD).
First, we prove that differentially private methods satisfy a "What You See Is What You Get (WYSIWYG)" generalization guarantee:
whatever a model does on its train data is almost exactly what it will do at test time.
This guarantee is formally captured by distributional generalization.
WYSIWYG enables principled algorithm design in deep learning
by reducing __generalization__ concerns to __optimization__ ones:
in order to mitigate unwanted behavior at test time,
it is provably sufficient to mitigate this behavior on the train data.
This is notably false for standard (non-DP) methods, hence this
observation has applications even when privacy is not required.
For example, importance sampling is known to fail for standard SGD,
but we show that it has exactly the intended effect for DP-trained models.
Thus, with DP-SGD, unlike with SGD, we can influence test-time behavior by
making principled train-time interventions.
We use these insights to construct simple algorithms which match or outperform SOTA in several distributional robustness applications, and to significantly improve the privacy vs. disparate impact trade-off of DP-SGD.
Finally, we also improve on known theoretical bounds relating differential privacy, stability, and distributional generalization.

---

## Setup

### Installation

Required packages and their versions are in [requirements.txt](requirements.txt).

### Dataset

Datasets are implemented in [wilds/datasets/archive/](wilds/datasets/archive/).

- To run UTKFace, download the dataset from https://github.com/aicip/UTKFace,
decompress the file in the `./data` directory, and rename the directory from
`UTKFace` to `UTKFace_v1.0`.

- To run the [adult dataset](https://archive.ics.uci.edu/ml/datasets/adult),
download the processed dataset from
https://github.com/spring-epfl/disparate-vulnerability/tree/main/data/adult
and save the file to `data/adult_v1.0`.

- Run `setup_datasets.py` to download multinli. Move the file into
`metadata_multinli.csv` into the folder `multinli` and move the folder
`multinli` to `data/multinli_v1.0`.

## Entry point

[examples/run_expt.py](examples/run_expt.py)

Set `--root_dir=./data` (if you follow the dataset setup in the previous section).

### Algorithms

- noisy gradient

Enable the `--apply_noise` flag.

- [SGD](examples/algorithms/ERM.py)

A sample command:

```bash
python examples/run_expt.py --algorithm ERM --optimizer SGD \
  --batch_size $BATCHSIZE
```

- DP-SGD

To run DP-IS-SGD, add the `--enable_privacy` flag and set DP related
arguments.

```bash
python examples/run_expt.py --algorithm ERM --optimizer SGD \
  --enable_privacy --sample_rate 0.0001 --delta 1e-5 \
  --max_per_sample_grad_norm 0.1
```

- DP-IS-SGD

To run DP-IS-SGD, add the `--weighted_uniform_iid` and `--enable_privacy` flag,
and set DP related arguments.

```bash
python examples/run_expt.py --algorithm ERM --optimizer SGD \
  --weighted_uniform_iid --enable_privacy --sample_rate 0.0001 --delta 1e-5 \
  --max_per_sample_grad_norm 0.1
```

- Importance Sampling SGD (IS-SGD)

To run DP-IS-SGD, add the `--uniform_over_groups` flag.

```bash
python examples/run_expt.py --algorithm ERM --optimizer SGD \
  --uniform_over_groups --batch_size $BATCHSIZE
```

- [IW-SGD](examples/algorithms/IWERM.py)

sample command:

```bash
python examples/run_expt.py --optimizer SGD --algorithm IWERM \
  --batch_size $BATCHSIZE
```

- [groupDRO](examples/algorithms/groupDRO.py)

To run [groupDRO](https://arxiv.org/abs/1911.08731), set `--algorithm groupDRO`.

```bash
python examples/run_expt.py --optimizer SGD --algorithm groupDRO \
  --batch_size $BATCHSIZE
```

- [DP-SGD-f](examples/algorithms/ERM_dpsgdf.py)

Set `--algorithm ERMDPSGDf`, `--enable_fair_privacy`, and other privacy related arguments.

```bash
python examples/run_expt.py --optimizer SGD --algorithm ERMDPSGDf \
  --max_per_sample_grad_norm $CLIPNORM --enable_fair_privacy \
  --batch_size $BATCHSIZE --delta 1e-5 --sigma ${SIGMA} \
  --uniform_iid --sample_rate $SAMPLERATE --C0 ${CLIPNORM} --sigma2 ${SIGMA2}
```


## Experiment Scripts

- [scrips/run_adult.sh](scrips/run_adult.sh)

- [scrips/run_celeba.sh](scrips/run_celeba.sh)
- [scrips/run_iNaturalist.sh](scrips/run_iNaturalist.sh)
- [scrips/run_utkface.sh](scrips/run_utkface.sh)
- [scrips/run_civilcomments.sh](scrips/run_civilcomments.sh)

### Citation

Code in this repository is modified from
[https://github.com/p-lambda/wilds](https://github.com/p-lambda/wilds).

For more experimental and technical details, please check our
[paper](https://arxiv.org/abs/2204.03230).
If any of our proposed algorithms or implementation is helpful, please cite:

```bibtex
@article{kulynych2022you,
  title={What You See is What You Get: Distributional Generalization for Algorithm Design in Deep Learning},
  author={Kulynych, Bogdan and Yang, Yao-Yuan and Yu, Yaodong and B{\l}asiok, Jaros{\l}aw and Nakkiran, Preetum},
  journal={arXiv preprint arXiv:2204.03230},
  year={2022}
}
```
