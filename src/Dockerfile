FROM nvidia/cuda:9.0-cudnn7-runtime-ubuntu16.04

WORKDIR /app

COPY inference/. /app
COPY requirements.txt /app/

RUN set -ex \
  && apt-get update \
  && apt-get -y upgrade \
  && apt-get -y install curl \
  && curl -sL https://deb.nodesource.com/setup_8.x | bash - \
  && apt-get -y install python3 python3-pip libgomp1 nodejs \
  && apt-get -y install libopencv-dev ffmpeg \
  && pip3 install --no-cache-dir -r requirements.txt \
  && rm requirements.txt \
  && apt-get -y remove python3-pip \
  && apt-get -y autoremove \
  && apt-get -y clean \
  && rm -rf /var/lib/apt/lists/* \
  && echo "Done!"


# RUN pip3 install -r requirements.txt

EXPOSE 7034

ENV MY APP

CMD ["python3", "main.py"]

