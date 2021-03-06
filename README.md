## vision-framework
The purpose of vision-framework is to develop a framework that can test various models, data, and loss functions by modifying only the config file.

### Dependencies
* Distributed gpu computing.
  + vision-framework is using [accelerate](https://github.com/huggingface/accelerate) for simple implementation of distributed gpu computing.
* Model
  + vision-framework includes several pretrained-model library such as [timm](https://fastai.github.io/timmdocs).
* Docker
  + All work was done with [pytorch:20.11-py3](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch), NGC docker.

### Requirements and Configuration
Install dependencies.
```
$ pip install -r requirments.txt
```
And you have to set configuration of accelerate.
```
$ accelerate config
```
All steps have been completed. Run the training code using accelerate
```
$ accelerate launch train.py --config-name=base
```

## To Do
* Add detection and segmentation and retrieval tasks