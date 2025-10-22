# Dataset Distillation by Matching Training Trajectories for Facial Recognition

## Facial Recognition
The initial MTT-Distillation worked with a set of generic objects, such as animals, objects, etc. 
This project investigates and builds on top of the original framework to see how MTT-Distillation interacts with facial data,
where CelebA dataset was selected as the baseline. The following deep learning models were used to teach the expert trajectories
throughout experimentation: MobileNetV2, VGGFace and EfficientNetB0.

The goal of this project was to see if MTT-Distillation could apply to specific applications,
in this case towards facial recognition.

### [Original MTT-Distillation Project Page](https://georgecazenavette.github.io/mtt-distillation/) | [Paper](https://arxiv.org/abs/2203.11932)
<br>

![Teaser image](docs/all_grid.png)

This repo contains code for training expert trajectories and distilling synthetic data from our Dataset Distillation by Matching Training Trajectories paper (CVPR 2022). Please see our [project page](https://georgecazenavette.github.io/mtt-distillation) for more results.


> [**Original Dataset Distillation by Matching Training Trajectories**](https://georgecazenavette.github.io/mtt-distillation/)<br>
> [George Cazenavette](https://georgecazenavette.github.io/), [Tongzhou Wang](https://ssnl.github.io/), [Antonio Torralba](https://groups.csail.mit.edu/vision/torralbalab/), [Alexei A. Efros](https://people.eecs.berkeley.edu/~efros/), [Jun-Yan Zhu](https://www.cs.cmu.edu/~junyanz/)<br>
> CMU, MIT, UC Berkeley<br>
> CVPR 2022 (Oral)

The task of "Dataset Distillation" is to learn a small number of synthetic images such that a model trained on this set alone will have similar test performance as a model trained on the full real dataset.

<img src='docs/method.gif' width=600>

Our method distills the synthetic dataset by directly optimizing the fake images to induce similar network training dynamics as the full,
real dataset. We train "student" networks for many iterations on the synthetic data,
measure the error in parameter space between the "student" and "expert" networks trained on real data,
and back-propagate through all the student network updates to optimize the synthetic pixels.

### Getting Started

First, clone repo:
```bash
git clone https://github.com/Thoemodachi/mtt-distillation.git
cd mtt-distillation
```

For an express instillation, we include ```.yaml``` files.

If you have an RTX 30XX GPU (or newer), run

```bash
conda env create -f requirements_11_3.yaml
```

If you have an RTX 20XX GPU (or older), run

```bash
conda env create -f requirements_10_2.yaml
```

You can then activate your conda environment with
```bash
conda activate distillation
```
##### Quadro Users Take Note:
```torch.nn.DataParallel``` seems to not work on Quadro A5000 GPUs, and this may extend to other Quadro cards.

If you experience indefinite hanging during training, try running the process with only 1 GPU by prepending ```CUDA_VISIBLE_DEVICES=0``` to the command.

### Generating Expert Trajectories
Before doing any distillation, you'll need to generate some expert trajectories using ```buffer.py```

The following command will train 100 MobileNet models on CelebA with ZCA whitening for 50 epochs each:
```bash
python buffer.py --dataset=celeba --model=MobileNetV2 --train_epochs=50 --num_experts=100 --buffer_path={path_to_buffer_storage} --data_path={path_to_dataset}
```

### Distillation by Matching Training Trajectories
The following command will then use the buffers we just generated to distill CelebA down to just 1 image per class:
```bash
python distill.py --dataset=celeba --ipc=1 --syn_steps=20 --expert_epochs=3 --max_start_epoch=20 --lr_img=0.1 --lr_lr=1e-05 --lr_teacher=0.01 --buffer_path={path_to_buffer_storage} --data_path={path_to_dataset}
```

<img src='docs/animation.gif' width=600>

Please find a full list of hyper-parameters below:

![image](https://user-images.githubusercontent.com/18726777/184226412-7bd0d577-225b-487c-8c9c-23f6462ca7d0.png)


## Acknowledgments
I would like to acknowledge the researchers who created the original MTT-Distillation codebase and conducted the initial research, George Cazenavette, Tongzhou Wang, Antonio Torralba, Alexei A. Efros, Jun-Yan Zhu. They have laid the foundation for the potential many applications of their form of dataset distillation, and their work was what inspired the initialisation of this project. The original MTT-Distillation code is adapted from https://github.com/VICO-UoE/DatasetCondensation and my code is adapted from https://github.com/GeorgeCazenavette/mtt-distillation.

## Related Work
<ol>
<li>
    Tongzhou Wang et al. <a href="https://ssnl.github.io/dataset_distillation/">"Dataset Distillation"</a>, in arXiv preprint 2018
</li>
<li>
    Bo Zhao et al. <a href="https://arxiv.org/abs/2006.05929">"Dataset Condensation with Gradient Matching"</a>, in ICLR 2020
</li>
<li>
    Bo Zhao and Hakan Bilen. <a href="https://arxiv.org/abs/2102.08259">"Dataset Condensation with Differentiable Siamese Augmentation"</a>, in ICML 2021
</li>
<li>
    Timothy Nguyen et al. <a href="https://arxiv.org/abs/2011.00050">"Dataset Meta-Learning from Kernel Ridge-Regression"</a>, in ICLR 2021
</li>
<li>
    Timothy Nguyen et al. <a href="https://arxiv.org/abs/2107.13034">"Dataset Distillation with Infinitely Wide Convolutional Networks"</a>, in NeurIPS 2021
</li>
<li>
    Bo Zhao and Hakan Bilen. <a href="https://arxiv.org/abs/2110.04181">"Dataset Condensation with Distribution Matching"</a>, in arXiv preprint 2021
</li>
<li>
    Kai Wang et al. <a href="https://arxiv.org/abs/2203.01531">"CAFE: Learning to Condense Dataset by Aligning Features"</a>, in CVPR 2022
</li>
</ol>

# Original MTT-Distillation Reference List
If you find our code useful for your research, please cite our paper.
```
@inproceedings{
cazenavette2022distillation,
title={Dataset Distillation by Matching Training Trajectories},
author={George Cazenavette and Tongzhou Wang and Antonio Torralba and Alexei A. Efros and Jun-Yan Zhu},
booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
year={2022}
}
```
