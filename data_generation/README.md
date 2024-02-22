# Data generation

## Preliminary

1. `pip install -r data_generation/requirements.txt`
2. Download the vqgan checkpoint from [CowTransfer](https://cowtransfer.com/s/d771c6d3d8344d) or [Google Drive](https://drive.google.com/drive/folders/1CyucT_QOArUH_Au8dfzRSwseyiCGserF?usp=share_link), and move it to `./weight/vqgan-f16-8192-laion`.

## Human keypoint

1. You can generate the keypoint image refer to [mmpose](https://mmpose.readthedocs.io/en/dev-1.x/demos.html#d-human-pose-estimation-with-inferencer) , and
   change the inference cmd like this

   ```shell
   python inferencer_demo.py data/path \
   coco/train2017/images \
   --pose2d configs/body_2d_keypoint/rtmo/coco/rtmo-l_16xb16-600e_coco-640x640.py \
   --pose2d-weights ./pth/rtmo-l_16xb16-600e_coco-640x640-516a421f_20231211.pth \
   --det-model demo/mmdetection_cfg/rtmdet_m_640-8xb32_coco-person.py \
   --black-background \
   --vis-out-dir coco/train2017/keypoints \
   --skeleton-style openpose \
   --disable-rebase-keypoint \
   --radius 8 \
   --thickness 4 \
   ```

2. Generate vq codebook by VQ-GAN

   ```shell
   python generate/generate_coco-keypoint.py \
   --input_data coco/train2017/images \
   --target_data coco/train2017/keypoints \
   --output_path vq_token/coco-keypoints/train2017
   ```

## Deblur

```shell
python generate/generate_GoPro.py \
--input_data GoPro_train/input \
--target_data GoPro_train/target \
--output_path vq_token/GoPro_train
```

## Derain

Here we use Rain13K data in lmdb fromat.

```shell
python generate/generate_Rain13K.py \
--input_data Rain13K_lmdb/input.lmdb \
--target_data Rain13K_lmdb/target.lmdb \
--output_path vq_token/Rain13K
```

## Video dataset

Here we use the HD-VILA-100M dataset.

1. You should download the dataset refer [hd-vila-100m](https://github.com/microsoft/XPretrain/tree/main/hd-vila-100m),
   and use [src/cut_videos.py](https://github.com/microsoft/XPretrain/blob/main/hd-vila-100m/src/cut_videos.py) to cut
   the videos to clips.

2. Generate vq codebook by VQ-GAN

   ```shell
   python generate/generate_hdvila_100m.py \
   --video_info_json hdvila_100m/cut_video_results/cut_part0.jsonl \
   --data_root hdvila_100m/video_clips_imgs \
   --output_root vq_token/hdvila_100m
   ```

## Segment mask

Here we use the SA-1B dataset.

1. Download the SA-1B dataset.

2. Generate vq codebook by VQ-GAN.

   ```shell
   python generate/generate_SA-1B.py \
   --tar_root SA-1B/tar \
   --img_json_root SA-1B/tmp/img_json \
   --mask_root SA-1B/tmp/mask \
   --output_path vq_token/SA-1B/token \
   --dp_mode
   ```