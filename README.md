
# VISION23 Challenge competition Track2: Synthetic data generation for defect detection


## Enviroment preparation
1. Please clone this repo by
    ```
    git clone https://github.com/vision-workshop/Track2.git
    cd Track2
    ```

2. Install [MMDetection](https://github.com/open-mmlab/mmdetection)

    ```
    conda create --name openmmlab python=3.8 -y
    conda activate openmmlab
    ```
    Please install the Pytorch corrersponding to your system version
    ```
    conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
    ```
    Then install MMDetection
    ```
    pip install -U openmim
    mim install mmcv-full
    pip install mmdet
    ```
3. Dataset Download

    Please download the datasets from here ```bit.ly/VISION_Datasets_Track_2``` and extract it to ```data``` folder.

4. Checkpoint preparation 

    Download pretrained check point by
    ```
    mkdir checkpoints
    cd checkpoints
    wget https://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_fpn_mstrain-poly_3x_coco/mask_rcnn_r50_fpn_mstrain-poly_3x_coco_20210524_201154-21b550bb.pth
    cd ..
    ```

5. Running the script on raw dataset

    Here we use  ```Console_sliced``` as an example
    ```
    python Block2.py --datasets Console_sliced
    ```
    The script will 
    1. Train a MaskRCNN model with Resnet50 Backbone on the ```train``` set
    2. Validate the model on the ```val``` set
    3. After training, the model weights stored in ```latest.pth``` will be used to conduct predition on the ```test``` set
    
    The training/validation log can be found under ```./work_dirs/Console_sliced/```. We provided a simple tool for training/validation log analysis named ```analysis.ipynb```.

    The prediction result will be stored as ```./results/Console_sliced.segm.json```

## Participant the competition
### Synthetic training data generation 
1. Please generate your own set of synthetic data, rename it as ```train``` dataset and replace the orginal ```train``` dataset under ```./data/Console_sliced/```. 
2. The generated synthetic images and annotations should strictly follow the format of the orginal training dataset ```train``` as follows: 

    This script expects datasets in the COCO format. Specifically, the datasets follow the layout below:

    ```
    dataset_name/
        train/
            _annotations.coco.json  # COCO format annotation
            0001.png                # Images
            0002.png
            ...
        val/
            _annotations.coco.json  # COCO format annotation
            0001.png                # Images
            0002.png
            ...
    ```
3. run the following code for the specific dataset (Please DO NOT change anything in the ```config.py``` or ```Block2.py``` file!)
    ```
    python Block2.py --datasets Console_sliced
    ``` 

### Result submission
The prediction result will be stored as ```./results/Console_sliced.segm.json```. Please compress ```Console_sliced.segm.json``` and upload to eval.ai. at 
```
https://eval.ai/web/challenges/challenge-page/1935/overview
```

## Rules:
1. You are only allowed to modify the ```train``` dataset. The training and testing script is fixed, please **DO NOT** change other files such as the ```config.py``` or ```Block2.py``` file! 
2. Data augmentation is allowed, the maximum number of augmentation of the dataset is 5x the original training set size.
3. Any synthetic data generation stretagy is welcome but any attempt to leak information from test set is strictly forbidden (we will check during the ```Test Phase```).
4. There are two phases in the challenge:
    - ```Dev phase```: you can submit your results multiple times and query test performance;
    - ```Test phase```: Generated dataset submission for verification purpose.
        - Please pay attention that you **ONLY** have two chances to submit your generated dataset.
        - Notice that the MaskRCNN may have small randomness, please run your dataset mutiple times to get a sense of the averarge performance. We will repeat 3 times and use the average value as the final score.
5. When submitting the generated dataset in the test phase, please also submit a technical report (at least 4 pages, following the CVPR tempalte at ```https://cvpr2023.thecvf.com/Conferences/2023/AuthorGuidelines```).
