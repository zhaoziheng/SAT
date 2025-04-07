FROM pytorch/pytorch:2.2.1-cuda11.8-cudnn8-runtime  # 或 python:3.9-slim

# 设置工作目录
WORKDIR /app

# 复制
COPY . /app

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt
RUN cd model
RUN pip install -e dynamic-network-architectures-main

# 指定容器启动命令
CMD ["python", "inference_cvpr25.py"]
