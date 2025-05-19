# Online CSLR Framework
Code for our proposed online continuous sign language recognition framework


## Data Preparation
For datasets and keypoint inputs, please check [../README.md](../README.md).

<br>
If you want to generate keypoint inputs yourself, please refer to the following steps:

Taking cls-daily as an example:
1. Obtain the text data from [CSL-Daily-text](https://github.com/FangyunWei/SLRT/tree/main/TwoStreamNetwork/data/csl-daily) . Obtain the video data from [CSL-Daily-video](http://home.ustc.edu.cn/~zhouh156/dataset/csl-daily/)
2. Put the text data into the folder ```../../data/cls-daily```, and check and modify the relevant configurations in the configuration file```configs/cls-daily_keypoint.yaml```
3. Keypoint generation
```
config_file='configs/cls-daily_keypoint.yaml'

python -m torch.distributed.launch --nproc_per_node 8 --master_port 29999 --use_env gen_pose.py --config=${config_file} --split=test
```

<br>
If you encounter an OOM (Out Of Memory) issue with storage, you can refer to the following code: slice the video frames, extract keypoints in batches according to the batch size, and then combine the results.

1. Slice the video frames

Split the video frames into smaller batches to reduce memory usage:
```
python preprocess.py --split test --filter_func prepare_text_data
```
2. Keypoint generation

Generate keypoints for each batch using the following command:
```
config_file='configs/ce_csl_keypoint.yaml'

python -m torch.distributed.launch --nproc_per_node 8 --master_port 29999 --use_env gen_pose.py --config=${config_file} --split=test
```
3. Merge results

After keypoint extraction, merge the results from all batches:
```
python preprocess.py --split test --filter_func merge_kp
```

### Segment Isolated Signs
We use a pre-trained CSLR model, TwoStream-SLR, to segment continuous sign videos into a set of isolated sign clips.
The pre-trained model checkpoints can be downloaded [here](https://github.com/FangyunWei/SLRT/blob/main/TwoStreamNetwork/docs/TwoStream-SLR.md)
After downloading, put them into the folder ``../../TwoStreamNetwork/result``

Next, navigate to [here](https://github.com/FangyunWei/SLRT/tree/main/TwoStreamNetwork) and execute the following commands to obtain the gloss_logits file.
```
config_file='experiments/configs/TwoStream/csl-daily_s2g.yaml'

python -u prediction.py --config=${config_file} --split dev,test
```

Then run
```
python gen_segment.py --datasetname csl --outputdir ../../data/csl-daily/ --only_blank --split dev,test

python gen_segment.py --datasetname csl --outputdir ../../data/csl-daily/ --split dev,test
```

### Sign Augmentation
Since the segmented signs are pseudo ground-truths and their boundaries may not be accurate, we further augment these segmented signs by running
```
python sign_augment.py --split test --datasetname csl
```
We provide processed meta data for [Phoenix-2014T](https://hkustconnect-my.sharepoint.com/:f:/g/personal/rzuo_connect_ust_hk/EtgOb0-NAWBHssQdx4zKj_IB7IA4mGk4Wuz5nRx0D8h5Bg?e=GqJYSp) and [CSL-Daily](https://hkustconnect-my.sharepoint.com/:f:/g/personal/rzuo_connect_ust_hk/Eu-Q1K-DlW1ChO2JjNBWXKsBN3otZ88z_RKXN9hEr5g9iA?e=uS6gbq).

## Training
```
config_file='configs/phoenix-2014t_ISLR.yaml'
python -m torch.distributed.launch --nproc_per_node 8 --master_port 29999 --use_env training.py --config=${config_file} --close_wandb
```
We provide model checkpoints for [Phoenix-2014T](https://hkustconnect-my.sharepoint.com/:f:/g/personal/rzuo_connect_ust_hk/EidJXFxpyaNPho5SKtVHEJ8BHex8Gq62koL-RrNnqtF1PA?e=IGGpxU) and [CSL-Daily](https://hkustconnect-my.sharepoint.com/:f:/g/personal/rzuo_connect_ust_hk/EhS5B3p9i3FNu5OpqFy3WyABkMMGg1VbAzMJrxjuFVOg6Q?e=c7OK0Z).


If you need to customize your training data and there are many blank segments, you can refer to the following code for data filtering:
```
python data_filter.py --split train --filter_func blank
```

## Testing (online inference)
```
config_file='configs/slide_phoenix-2014t.yaml'
python -m torch.distributed.launch --nproc_per_node 1 --master_port 29999 --use_env prediction_slide.py --config=${config_file}  --save_fea 1
```
The flag "--save_fea" is optional, which aims to extract features for boosting an offline model with the well-optimized online model.