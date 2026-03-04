FROM python:3.11.9

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN python -m nltk.downloader stopwords wordnet

COPY flask_app/ /app/flask_app/
COPY models/vectorizer.pkl /app/models/vectorizer.pkl

EXPOSE 5000

CMD ["gunicorn","-b","0.0.0.0:5000","flask_app.app:app"]


