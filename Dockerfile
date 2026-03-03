FROM python:3.11-slim

# libgomp1 needed by LightGBM
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

# HF Spaces requires user ID 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH
WORKDIR $HOME/app

COPY --chown=user v15/panel_dashboard/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user . .

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=$HOME/app

EXPOSE 7860

CMD ["panel", "serve", "v15/panel_dashboard/app.py", \
     "--address", "0.0.0.0", "--port", "7860", \
     "--allow-websocket-origin", "*", \
     "--num-procs", "1"]
