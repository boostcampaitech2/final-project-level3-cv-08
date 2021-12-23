FROM python:3.7

WORKDIR /app

COPY requirements.txt ./requirements.txt

RUN python -m pip install --upgrade pip

RUN apt-get update

RUN  apt-get -y install libgl1-mesa-glx

RUN pip3 install -r requirements.txt

EXPOSE 8080

COPY . /app

CMD streamlit run --server.port 8080 --server.enableCORS false app.py