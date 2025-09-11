FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml .

RUN pip install poetry
RUN poetry install --no-root

COPY . .
RUN poetry run python train.py

CMD ["/bin/bash"]