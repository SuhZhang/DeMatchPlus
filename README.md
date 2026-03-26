# DeMatchPlus
Pytorch implementation of DeMatch++ for a comprehensive extension of CVPR'24 paper ["DeMatch: Deep Decomposition of Motion Field for Two-View Correspondence Learning"](https://openaccess.thecvf.com/content/CVPR2024/html/Zhang_DeMatch_Deep_Decomposition_of_Motion_Field_for_Two-View_Correspondence_Learning_CVPR_2024_paper.html).

Accepted by TPAMI 2025, ["DeMatch++: Two-View Correspondence Learning via Deep Motion Field Decomposition and Respective Local-Context Aggregation"](https://ieeexplore.ieee.org/abstract/document/11119297).

This repo contains the code and data for essential matrix estimation.

## Requirements

Please use Python 3.6, opencv-contrib-python (3.4.0.12) and Pytorch (>= 1.7.0). Other dependencies should be easily installed through pip or conda.


## Example scripts

### Run the demo

For a quick start, clone the repo and download the pretrained model.
```bash
git clone https://github.com/SuhZhang/DeMatchPlus 
cd DeMatchPlus 
```
Then download the pretrained models from [here](https://drive.google.com/drive/folders/1OK3jq9SM5iUrK_rOebwi5FjWXs99tr3k?usp=drive_link).

Then run the feature matching with demo.

```bash
cd ./demo && python demo.py
```


### Test pretrained model

We provide the model trained on YFCC100M and SUN3D. Run the test script to get results (the generated putative matches can be downloaded from [here](https://drive.google.com/drive/folders/1utkm7K1w9vy02HVzQ6PFf3k5OXUDkHdf?usp=drive_link)).

```bash
cd ./test 
python test.py
```
You can change the default settings for test in `./test/config.py`.

