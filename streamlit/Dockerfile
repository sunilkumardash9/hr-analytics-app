FROM python:3.9.15-slim-bullseye


WORKDIR /streamlit_code

COPY requirements.txt /streamlit_code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /streamlit_code/requirements.txt 

COPY ./streamlit-app.py ./office.jpg /streamlit_code/

EXPOSE 8501

CMD [ "streamlit", "run", "streamlit-app.py"]
