# 硬件解码配合TensorRT
- 配置tensorRT一样的环境
- 增加NVDEC和ffmpeg的配置
- `make yolo -j64`
- 软解码和硬解码，分别消耗cpu和gpu资源。在多路，大分辨率下体现明显
- 硬件解码和推理可以允许跨显卡
- 理解并善于利用的时候，他才可能发挥最大的效果