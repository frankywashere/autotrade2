FROM python:3.11-slim

WORKDIR /app

COPY v15/panel_dashboard/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1

EXPOSE 7860

CMD ["panel", "serve", "v15/panel_dashboard/app.py", \
     "--address", "0.0.0.0", "--port", "7860", \
     "--allow-websocket-origin", "*", \
     "--num-procs", "1"]
