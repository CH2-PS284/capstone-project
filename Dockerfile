FROM python:3.9-slim

ENV PYTHONUNBUFFERED True
ENV APP_HOME /app 
WORKDIR $APP_HOME
COPY . ./

RUN apt-get update -y && apt-get install -y \
  build-essential cmake \
  && apt-get clean


RUN pip install -r requirements.txt

ENTRYPOINT [ "/usr/local/bin/gunicorn", "--config", "gunicorn.conf.py", "--log-config", "gunicorn-logging.conf", "hellomodule.app:app" ]
