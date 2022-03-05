FROM continuumio/miniconda3

RUN apt-get update
RUN apt-get install unzip

RUN git clone https://github.com/jrwnter/cddd.git 
WORKDIR cddd/
RUN git checkout 25c1d6f78d9e766f30d04b97e23cb75464785c04

RUN conda env create -f environment.yml
RUN echo "conda activate cddd" >> ~/.bashrc
SHELL ["/bin/bash", "--login", "-c"]

RUN pip install gdown && gdown 1oyknOulq_j0w9kzOKKIHdTLo5HphT99h && unzip default_model.zip -d cddd/data && rm default_model.zip
RUN pip install tensorflow==1.10

COPY requirements.txt . 
RUN pip install -r requirements.txt && rm requirements.txt

EXPOSE 80
COPY main.py .
COPY utils.py .

CMD ["/opt/conda/envs/cddd/bin/uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
