# Enhancing Stable Video Diffusion Prompts

![overview gif](cheetah.gif) 

> Example article: `In Africa's vast savannah, a swift cheetah races, epitomizing nature's splendor. Its effortless sprint highlights not just its remarkable speed but also the urgent need to protect these majestic creatures and their diminishing habitats.`
> 
> Processed prompts: `best quality,ultra-detailed,masterpiece,hires,8k,In Africa,vast savannah,swift cheetah,races,epitomizing,nature,splendor,effortless,sprint,highlights,remarkable speed,also,urgent need,protect,majestic creatures,diminishing,habitats`

## Prerequirement

Python 3.10.0

### PyTorch

Based on individual OS environments, please download corresponding binary files: https://pytorch.org/get-started/locally/

## Install

```sh
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -m nltk.downloader stopwords
```

<!---
### Conda

```sh
conda env create -f requirement.yaml
conda env list
python -m spacy download en_core_web_sm
python -m nltk.downloader stopwords
```
--->
