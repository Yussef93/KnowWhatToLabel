# KnowWhatToLabel
This is the official repository for the paper of "Knowing What to Label for Few Shot Microscopy Image Cell Segmentation" accepted to IEEE Winter Applications on Computer vision (WACV) 2023. 

## Abstract
 In microscopy image cell segmentation, it is common to train a deep neural network on source data, containing different types of microscopy images, and then fine-tune it using a support set comprising a few randomly selected and annotated training target images. In this paper, we argue that the random selection of unlabelled training target images to be annotated and included in the support set may not enable an effective fine-tuning process, so we propose a new approach to optimise this image selection process. Our approach involves a new scoring function to find informative unlabelled target images. In particular, we propose to measure the consistency in the model predictions on target images against specific data augmentations. However, we observe that the model trained with source datasets does not reliably evaluate consistency on target images. To alleviate this problem, we propose novel self-supervised pretext tasks to compute the scores of unlabelled target images. Finally, the top few images with the least consistency scores are added to the support set for oracle (i.e., expert) annotation and later used to fine-tune the model to the target images. In our evaluations that involve the segmentation of five different types of cell images, we demonstrate promising results on several target test sets compared to the random selection approach as well as other selection approaches, such as Shannon's entropy and Monte-Carlo dropout. 

## Requirements

### Step 1) Create environment and download datasets

Create enviornment using conda and "env.yml". Download the datasets manually and place them in 'Datasets/FewShot/Raw/' in the following format

```
B5/
├── Image/
├── Groundtruth/
├── Groundtruth_PL/

B39/
├── Image/
├── Groundtruth/
├── Groundtruth_PL/

TNBC/
├── Image/
├── Groundtruth/
├── Groundtruth_PL/


EM/
├── Image/
├── Groundtruth/
├── Groundtruth_PL/


ssTEM/
├── Image/
├── Groundtruth/
├── Groundtruth_PL/
```
Copy and paste contents of Groundtruth/ to Groundtruth_PL/

### Step 2) Download Models trained on Pseudo-label cell segmentation 
Place pre-trained models using this link https://cloudstore.uni-ulm.de/s/LRmy6XcDw57EXCx and place them inside the working directory.

### Step 3) Run code (Selection and fine-tuning)
```
python evaluation_main.py --target $TARGET_DATASET --num_shot $K_SHOT --select 'Ours'
```

## Cite
```
@inproceedings{dawoud2023knowing,
  title={Knowing What to Label for Few Shot Microscopy Image Cell Segmentation},
  author={Dawoud, Youssef and Bouazizi, Arij and Ernst, Katharina and Carneiro, Gustavo and Belagiannis, Vasileios},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages={3568--3577},
  year={2023}
}
```
