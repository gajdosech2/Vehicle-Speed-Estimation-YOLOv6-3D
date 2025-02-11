# YOLOv6 Transform 3D

This repository is based on YOLOv6 implementation by Meituan [https://github.com/meituan/YOLOv6](https://github.com/meituan/YOLOv6).
The models predict 2D bounding box with additional parameters for 3D lifting. These changes are documented in our paper and were also described in the paper [Detection of 3D Bounding Boxes of Vehicles Using Perspective Transformation for Accurate Speed Measurement](https://arxiv.org/pdf/2003.13137.pdf).

## Dataset

To obtain BrnoCompSpeed dataset, you need to contact authors, as it isn't public. Contact information can be found [here](https://github.com/JakubSochor/BrnoCompSpeed).

## Requirements

To use the models and run our whole pipeline, please install the requirements first:

``pip install -r requirements.txt``

We have tested the code on:
Nvidia driver: **535.54.03**  
CUDA version: **12.2**

## Training the Detector

To train our modified YOLOv6 model, you need to prepare your dataset to the YOLO format. Example of directories and files structure of YOLO format:

```
custom_dataset
├── images
│   ├── train
│   │   ├── train0.jpg
│   │   └── train1.jpg
│   └── val
│       ├── val0.jpg
│       └── val1.jpg
└── labels
 ├── train
 │   ├── train0.txt
 │   └── train1.txt
 └── val
 ├── val0.txt
 └── val1.txt
```

Example of annotation file in YOLO format:

```txt
# class_id center_x center_y bbox_width bbox_height
0 0.300926 0.617063 0.601852 0.765873
```  

After creating the dataset, you need to create a file ``custom.yaml`` in ``./data/`` directory.

``
python ./yolov6/train.py <args>
``  
All arguments can be found in ``./yolov6/train.py`` file. During the training process, the best model is saved in ``./runs/train/<experiment_name>/weights/best.pt``.

## Testing Speed Evaluation Pipeline

After obtaining BrnoCompSpeed dataset, you need to place the dataset in the format as:

```
 ├── BrnoCompSpeed
 │    ├── dataset
 │    │      ├── season0_center
 │    │      │     ├── video.avi
 │    │      │     └── video_mask
 │    │      ...
 │    │      └── season6_right
 │    │            ├── video.avi
 │    │            └── video_mask
 │    │              
 │    └── results
 │          ├── season0_center
 │          │     └── system_SochorCVIU_Edgelets_BBScale_Reg.json # Make sure to include the calibration file, without it, the pipeline will not work 
 │          ...
 │          └── season6_right
 │                └── system_SochorCVIU_Edgelets_BBScale_Reg.json #    
```

All paths and arguments need to be changed in ``speed_estimation.py`` or in ``tensorrt_estimation.py`` for TensorRT INT8 quantization.

Alternatively, you can run our pipeline using command line arguments:
``python speed_estimation.py --weights=<path_to_the_checkpoint> --test-name=<name_of_the_test> \\  
--batch-size-processing=32 --half --root_dir_video_path=<path_to_dataset> --root_dir_results_path=<path_to_results>``

If you want to change input size and/or other properties, look for all arguments in the corresponding script files.

## Quantization and TensorRT

To be able to run quantization and TensorRT, you need to install CUDA Toolkit [here](https://developer.nvidia.com/cuda-downloads), and of course, you need to have compatible hardware. To learn more about TensorRT and quantization, look at [this](https://developer.nvidia.com/blog/achieving-fp32-accuracy-for-int8-inference-using-quantization-aware-training-with-tensorrt/) blog post.

1. The checkpoint obtained from training needs to be exported to ONNX format:
   ``python ./deploy/ONNX/export_onnx.py --weights=<path_to_the_model> --img-size=<size_of_input_image>``
 All arguments can be found in ``./deploy/ONNX/export_onnx.py`` file.
2. After exporting the model to ONNX format, you need to run quantization:
   ``python ./deploy/TensorRT/onnx_to_trt.py --model=<path_to_the_ONNX_model> --dtype int8 --verbose --calib-img-dir=<path_to_calibration_data>``
 All arguments can be found in ``./deploy/TensorRT/onnx_to_trt.py`` file. To run quantization, you need to provide calibration data. 
3. To run the pipeline on video utilizing TensorRT with INT8 operation precision:
   ``python tensorrt_estimation.py --trt-model=<path_to_the_trt_model> --test-name=<name_of_the_test> --root_dir_video_path=<path_to_dataset> --root_dir_results_path=<path_to_results>\\``

## Evaluation of Speed Estimation

All our trained models on BrnoCompSpeed dataset are in the ``checkpoints`` directory (large model exceeds GitHub's file size limit of 100.00 MB) or alternatively can be downloaded
from [here](https://drive.google.com/file/d/1l729UFXxHdzzTf1yJuJcpYbtrqPEIfPh/view?usp=sharing).

To evaluate speed estimation, you need to use ``eval.py`` from [BrnoCompSpeed/code](https://github.com/JakubSochor/BrnoCompSpeed/tree/master/code) repository.

