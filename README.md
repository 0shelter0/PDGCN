
# Posed Mixed Dynamic Graph Convolution Network for Group Activity Recognition

This is an official implementation of *Posed Mixed Dynamic Graph Convolution Network for Group Activity Recognition*, namely shelterX's Degree M.S. Thesis. In this repo, we open the PyTorch-based codebase corresponding to training and inference as described in the Thesis. The overall architecture of PDGCN is shown as below.

![overall framwork of PDGCN](figures/2-branch_crop.pdf)

## Dependencies

- OS: Linux (CentOS 7)
- GPU: NVIDIA Tesla V100
- Python: `3.7`
- CUDA: `10.2`
- PyTorch: `1.10`
- Torchvision: `0.11.2`
- [RoIAlign for Pytorch](https://github.com/longcw/RoIAlign.pytorch)

## Prepare Datasets

- Option one
The steps following refer to the DIN official reop [https://github.com/JacobYuan7/DIN-Group-Activity-Recognition-Benchmark](https://github.com/JacobYuan7/DIN-Group-Activity-Recognition-Benchmark).

    1. Download publicly available datasets from following links: [Volleyball dataset](http://vml.cs.sfu.ca/wp-content/uploads/volleyballdataset/volleyball.zip) and [Collective Activity dataset](http://vhosts.eecs.umich.edu/vision//ActivityDataset.zip).
    2. Unzip the dataset file into `data/volleyball` or `data/collective`.
    3. Download the file `tracks_normalized.pkl` from [cvlab-epfl/social-scene-understanding](https://raw.githubusercontent.com/wjchaoGit/Group-Activity-Recognition/master/data/volleyball/tracks_normalized.pkl) and put it into `data/volleyball/videos`

- Option two:
For the convenience of reproducibility in the Thesis, you can refer the repo below to prepare the using datasets more easily:
  - [Dataset Preparation](https://github.com/hongluzhou/composer#dataset-preparation)
  - Composer offical repo: [https://github.com/hongluzhou/composer](https://github.com/hongluzhou/composer)

## Get Started

1. **Train the Base Model**: Fine-tune the base model for the dataset.

    ```shell
    # Volleyball dataset
    cd PROJECT_PATH 
    python scripts/train_volleyball_stage1.py
    
    # Collective Activity dataset
    cd PROJECT_PATH 
    python scripts/train_collective_stage1.py
    ```

2. **Train with the reasoning module**: Append the reasoning modules onto the base model to get a reasoning model.
    1. **Volleyball dataset**
        - **DIN** 
            ```
            python scripts/train_volleyball_stage2_dynamic.py
            ```
       - **lite DIN** \
            We can run DIN in lite version by setting *cfg.lite_dim = 128* in *scripts/train_volleyball_stage2_dynamic.py*.
            ```
            python scripts/train_volleyball_stage2_dynamic.py
            ```
       - **ST-factorized DIN** \
            We can run ST-factorized DIN by setting *cfg.ST_kernel_size = [(1,3),(3,1)]* and *cfg.hierarchical_inference = True*.
        
            **Note** that if you set *cfg.hierarchical_inference = False*, *cfg.ST_kernel_size = [(1,3),(3,1)]* and *cfg.num_DIN = 2*, then multiple interaction fields run in parallel.
            ```
            python scripts/train_volleyball_stage2_dynamic.py
            ```
        
        Other model re-implemented by us according to their papers or publicly available codes:
        - **AT** 
            ```
            python scripts/train_volleyball_stage2_at.py
            ```
        - **PCTDM** 
            ```
            python scripts/train_volleyball_stage2_pctdm.py
            ```
        - **SACRF** 
            ```
            python scripts/train_volleyball_stage2_sacrf_biute.py
            ```
       - **ARG** 
            ```
            python scripts/train_volleyball_stage2_arg.py
            ```
        - **HiGCIN** 
            ```
            python scripts/train_volleyball_stage2_higcin.py
            ```
       
    2. **Collective Activity dataset**
        -  **DIN** 
            ```
            python scripts/train_collective_stage2_dynamic.py
            ```
        -  **DIN lite** \
        We can run DIN in lite version by setting 'cfg.lite_dim = 128' in 'scripts/train_collective_stage2_dynamic.py'.
            ```
            python scripts/train_collective_stage2_dynamic.py
            ```

Another work done by us, solving GAR from the perspective of incorporating visual context, is also [available](https://ojs.aaai.org/index.php/AAAI/article/view/16437/16244).
```bibtex
@inproceedings{yuan2021visualcontext,
  title={Learning Visual Context for Group Activity Recognition},
  author={Yuan, Hangjie and Ni, Dong},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={35},
  number={4},
  pages={3261--3269},
  year={2021}
}
```







