FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7
COPY requirements.txt .

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

RUN pip install -r requirements.txt
RUN pip install pandas
COPY ./app /app
