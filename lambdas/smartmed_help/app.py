"""
Lambda: smartmed_help — chat with Quiz Mode and Pharmacist Summary
"""
import json, os, boto3
from flask import Flask, request, Response
from mangum import Mangum

flask_app = Flask(__name__)
bedrock   = boto3.client("bedrock-runtime", region_name=os.environ.get("AWS_REGION_NAME", "ap-southeast-5"))
MODEL     = "global.anthropic.claude-haiku-4-5-20251001-v1:0"

CORS_HEADERS = {
    "Access-Control-Allow-Origin":  "*",
    "Access-Control-Allow-Headers": "Content-Type",
    "Access-Control-Allow-Methods": "POST,OPTIONS",
}

SYSTEM_PROMPT = (
    "You are SmartMed Help. Use emojis to make answers easy to read.\n\n"
    "CORE RULES:\n"
    "- Treat all inputs as patient-reported or AI-extracted draft.\n"
    "- Be brief — max 100 words unless more genuinely helps.\n"
    "- Do not diagnose or recommend starting, stopping, or changing medicines.\n\n"
    "EMERGENCY: If user describes severe allergic reaction, overdose, chest pain, or difficulty breathing:\n"
    "🚨 In Malaysia: call 999 now. Outside Malaysia: call your local emergency number.\n\n"
    "QUIZ MODE (trigger: quiz me):\n"
    "- Up to 3 questions, one at a time. Score at end.\n\n"
    "PHARMACIST SUMMARY (trigger: prepare my pharmacist summary):\n"
    "Generate a copyable plain-text block with medicines, instructions, allergies, other medicines, and 3 questions to ask."
)


@flask_app.route("/", methods=["OPTIONS"])
def options():
    return Response("", status=200, headers=CORS_HEADERS)


@flask_app.route("/", methods=["POST"])
def handler():
    body    = request.get_json(force=True, silent=True) or {}
    lang    = body.get("preferred_language", "English")
    card    = body.get("medicine_card", "")
    precautions = body.get("my_medicine_summary", "")
    history = body.get("history", [])
    message = body.get("message", "Hello")

    system = (
        f"{SYSTEM_PROMPT}\n\n"
        f"Respond in {lang}.\n\n"
        f"Medicine Card: {card or '(none)'}\n\n"
        f"Precautions: {precautions or '(none)'}"
    )

    messages = []
    for turn in history:
        if turn.get("role") in ("user", "assistant") and turn.get("content"):
            messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": message})

    resp = bedrock.invoke_model(
        modelId=MODEL,
        contentType="application/json",
        accept="application/json",
        body=json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1024,
            "system": system,
            "messages": messages,
        }),
    )
    result = json.loads(resp["body"].read())
    text   = result.get("content", [{}])[0].get("text", "")
    return Response(text, status=200, content_type="text/plain; charset=utf-8", headers=CORS_HEADERS)


app = Mangum(flask_app)
