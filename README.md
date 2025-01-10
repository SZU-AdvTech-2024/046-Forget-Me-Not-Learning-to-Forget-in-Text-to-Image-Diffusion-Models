# Forget-Me-Not: Learning to Forget in Text-to-Image Diffusion Models
#### Features

- Forget-Me-Not is a plug-and-play, efficient and effective concept forgetting and correction method for large-scale text-to-image models.
- It provides an efficient way to forget specific concepts with as few as 35 optimization steps, which typically takes about 30 seconds.
- It can be easily adapted as lightweight patches for Stable Diffusion, allowing for multi-concept manipulation and convenient distribution.
- Novel attention re-steering loss demonstrates that pretrained models can be further finetuned solely with self-supervised signals, i.e. attention scores.

## Setup

```
conda create -n forget-me-not python=3.8
conda activate forget-me-not

pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116

pip install -r requirements.txt
```

## Train
We provde an example of forgetting the identity of Elon Musk.

- First, train Ti of a concept. Optional, only needed if `use_ti: true` in `attn.yaml`.
```
python run.py configs/ti.yaml
```
- Second, use attention resteering to forget a concept.
```
python run.py configs/attn.yaml
```
- Results can be found in `exps_ti` and `exps_attn`.

## Empirical Guidance

- Modify `ti.yaml` to tune Ti. In practical, prompt templates, intializer tokens, the number of tokens all have influences on inverted tokens, thus affecting forgetting results.
- Modify `attn.yaml` to tune forgetting procedure. Concept and its type are specified under `multi_concept` as `[elon-musk, object]`. During training, `-` will be replaced with space as the plain text of the concept. A folder containing training images are assumed at `data` folder with the same name `elon-musk`. Set `use_ti` to use inverted tokens or plain text of a concept. Set `only_optimize_ca` to only tune cross attention layers. otherwise UNet will be tuned. Set `use_pooler` to include pooler token `<|endoftext|>` into attention resteering loss.
- To achieve the best results, tune hyperparameters such as `max_train_steps` and `learning_rate`. They can vary concept by concept.
- Use precise attention scores could be helpful, e.g. instead of using all pixels, segment out the pixels of a face, only using their attention scores for forgetting an identity.
