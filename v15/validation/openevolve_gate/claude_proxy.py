"""
Flask proxy that routes OpenEvolve API calls to Claude Code CLI.
Run on the Windows server alongside OpenEvolve.

Uses stdin instead of CLI args to avoid Windows ~32K command-line limit.
"""

import json
import subprocess
import sys
import tempfile
import time
import os
from flask import Flask, request, jsonify

app = Flask(__name__)

# Path to Claude Code CLI on Windows
CLAUDE_CMD = r"C:\Users\frank\.local\bin\claude.exe"
MODEL = "haiku"
PORT = 5563


@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    data = request.json
    messages = data.get('messages', [])

    # Build prompt from messages
    prompt_parts = []
    for msg in messages:
        role = msg.get('role', 'user')
        content = msg.get('content', '')
        if role == 'system':
            prompt_parts.append(f"System: {content}")
        elif role == 'user':
            prompt_parts.append(content)

    full_prompt = "\n\n".join(prompt_parts)

    try:
        # Use stdin to pass prompt — avoids Windows command-line length limit
        result = subprocess.run(
            [
                CLAUDE_CMD,
                "--print",
                "--model", MODEL,
                "--dangerously-skip-permissions",
            ],
            input=full_prompt,
            capture_output=True,
            text=True,
            timeout=300,
            encoding='utf-8',
        )
        response_text = result.stdout.strip()
        if not response_text:
            response_text = result.stderr.strip() or "No output from claude"
    except subprocess.TimeoutExpired:
        response_text = "Error: claude timed out after 300s"
    except Exception as e:
        response_text = f"Error: {str(e)}"

    return jsonify({
        "id": f"chatcmpl-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": f"claude-{MODEL}",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": response_text,
            },
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    })


@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"})


if __name__ == '__main__':
    port = int(sys.argv[1]) if len(sys.argv) > 1 else PORT
    print(f"Starting Claude proxy on port {port} (model: {MODEL})...")
    app.run(host='0.0.0.0', port=port, debug=False)
