FROM python:3.11

RUN python -m pip install --upgrade pip tox

RUN apt-get update && DEBIAN_FRONTEND=noninteractive && apt-get install -y --no-install-recommends \
   libgl1-mesa-glx \
&& rm -rf /var/lib/apt/list/*

COPY . /app
WORKDIR /app

RUN pip install --no-cache-dir .

CMD ["tox", "-e", "test"]
