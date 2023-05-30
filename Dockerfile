FROM pytorch/pytorch:1.12.0-cuda11.3-cudnn8-runtime
RUN apt-get update && apt install -y git
RUN pip install tensorboardX
RUN pip install ttach
RUN pip install pandas
RUN pip install matplotlib
RUN pip install opencv-python
RUN pip install google-cloud-storage