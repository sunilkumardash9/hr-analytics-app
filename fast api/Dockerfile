FROM python:3.9.15-slim-bullseye as build

WORKDIR /code

COPY .requirements.txt ./code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt 

EXPOSE 8000

COPY ./hr_analytics_api.py ./my-model2 /code/

CMD ['python', 'hr_analytics_api.py']

