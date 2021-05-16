#!/bin/bash
docker run -it --gpus all -v /home/cadu/Nextcloud/Projects/OEMC:/workspace --device /dev/nvidia0 --device /dev/nvidia-uvm --device /dev/nvidiactl pytorch/pytorch
