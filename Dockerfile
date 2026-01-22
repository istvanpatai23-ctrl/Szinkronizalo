FROM python:3.9.13-slim

RUN apt-get update && apt-get install -y ffmpeg espeak-ng && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Másoljuk be a letöltött csomagokat a konténerbe
COPY pkgs/ /app/pkgs/

# Telepítés a helyi fájlokból
RUN pip install --no-cache-dir /app/pkgs/*.whl /app/pkgs/*.tar.gz

COPY . .
EXPOSE 7860
CMD ["python", "app.py"]
