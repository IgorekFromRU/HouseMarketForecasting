FROM python:3.9
WORKDIR /usr/src/app
COPY ./app/ ./
COPY ./requirements.txt ./
# RUN python -m pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install xgboost
RUN apt-get clean
CMD [ "python", "./server.py" ]