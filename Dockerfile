FROM python:3.6

WORKDIR DEZERO
COPY . .

RUN pip install -r requirements.txt