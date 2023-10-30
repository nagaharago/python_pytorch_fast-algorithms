FROM python:3.9-buster

WORKDIR /workspace

RUN apt-get -y update

EXPOSE 8501

COPY /requirements.txt /requirements.txt
RUN pip install -U pip && \
    pip install --no-cache-dir -r /requirements.txt
