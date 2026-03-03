FROM python:3.11-slim

WORKDIR /app

# libgomp1 needed by LightGBM
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

COPY v15/panel_dashboard/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

EXPOSE 7860

CMD ["panel", "serve", "v15/panel_dashboard/app.py", \
     "--address", "0.0.0.0", "--port", "7860", \
     "--allow-websocket-origin", "*", \
     "--num-procs", "1"]
