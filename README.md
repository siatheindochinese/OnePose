# OnePose (but with CAD Models)
But why?

Because I want my SfM point-cloud model to be properly aligned with my CAD Model canonically.

But why?

I want to render stuff relative to the canonical pose of my CAD model.

Added pipeline for realtime visualization.

### [Original Project Page](https://zju3dv.github.io/onepose) | [Paper](https://arxiv.org/pdf/2205.12257.pdf)
<br/>

> OnePose: One-Shot Object Pose Estimation without CAD Models  
> [Jiaming Sun](https://jiamingsun.ml)<sup>\*</sup>, [Zihao Wang](http://zihaowang.xyz/)<sup>\*</sup>, [Siyu Zhang](https://derizsy.github.io/)<sup>\*</sup>, [Xingyi He](https://github.com/hxy-123/), [Hongcheng Zhao](https://github.com/HongchengZhao), [Guofeng Zhang](http://www.cad.zju.edu.cn/home/gfzhang/), [Xiaowei Zhou](https://xzhou.me)   
> CVPR 2022  

![demo_vid](assets/realtimebysia.gif)

Refer to the original repository for setup instructions.

## IMPORTANT
Modified for private use-case.

- Added YoloV5 detector, weights located at data/yolov5/
- Added realtime scripts cv2_real_time_improved.py and cv2_real_time_improved.sh
- Added a modified feature-matching object detector (modified for realtime, but it still sucks lmao)

## Acknowledgement
This repository uses work from [YoloV5](https://github.com/ultralytics/yolov5) and [OnePose](https://github.com/zju3dv/OnePose), neither of which I am originally involved in.
