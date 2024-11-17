FROM python:3.9

WORKDIR /app

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt
RUN python -c "import nltk;nltk.download('stopwords');nltk.download('punkt_tab')"

COPY joblib_data/. .


COPY utilities.py .
COPY model.py .
COPY main.py .


CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]