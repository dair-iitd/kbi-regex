# KBI-Regex

This repository contains code and data for reproducing the results of our paper:

Vaibhav Adlakha, Parth Shah, Srikanta Bedathur, Mausam. [Regex Queries over Incomplete Knowledge Bases](https://www.akbc.ws/2021/papers/4YQVfA5vEJS)

To cite this work, please use the following citation:
```
@inproceedings{
adlakha2021regex,
title={Regex Queries over Incomplete Knowledge Bases},
author={Vaibhav Adlakha and Parth Shah and Srikanta J. Bedathur and Mausam .},
booktitle={3rd Conference on Automated Knowledge Base Construction},
year={2021},
url={https://openreview.net/forum?id=4YQVfA5vEJS}
}
```

## Downloading Datasets
The paper introduces two datasets for regex queries over incomplete knowledge bases - FB15K-Regex and Wiki100-Regex. These datasets are available to be downloaded from the following links - 
1. [FB15K-Regex](https://zenodo.org/record/7071856/files/fb15k_regex_data.zip)
2. [Wiki100-Regex](https://zenodo.org/record/7071856/files/wiki_v2_regex_data.zip)

Each dataset is a `.zip` file. This needs to be placed in a `cache/raw-datasets` directory within the project directory. The directory structure should look like the following:
```
kbi-regex
     |___ cache
            |___ raw-datasets
                     |____ fb15k_regex_data.zip
                     |____ wiki_v2_regex_data,zip
```
The zip files will be unpacked and encoded when the code is run for the first time, and will be used for subsequent runs.


## Running Models
To run the model, we first need to install the repository as a package. From the project directory, run the following command:
```
pip install --no-index --no-deps -e .
```

Use the following command to run RotatE-Box (Free parameter + Aggregation) on FB15K-Regex
```
python main.py \
    --model RotatE --box \
    --kleene_plus_op free_param \
    --dataset fb15k \
    -dim 800 \
    --query_types "0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 21" \
    --do_train \
    --learning_rate 1e-4 \
    --lr_schedule half \
    --truncate_loss \
    --batch_size 1024 \
    --negative_sample_count 256 \
    --max_epochs 500 \
    --resume_from_checkpoint <PATH TO KBC MODEL> \
    --seed 0 \
    --num_workers 8 \
    --save_dir <PATH TO OUTPUT DIR>
```
## Contact
For queries and clarifications please contact **vaibhav.adlakha (at) mila (dot) quebec**
