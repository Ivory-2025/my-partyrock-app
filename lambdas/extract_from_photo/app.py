"""
Lambda: extract_from_photo
Native Lambda handler — buffered response via Mangum + Flask.
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
    "You are a clinical pharmacist assistant. A patient has uploaded a photo or scan of a medicine label. "
    "Extract the details and display them field by field.\n\n"
    "RULES:\n"
    "- Extract only what is explicitly visible. Do not infer missing fields.\n"
    "- If a field is not visible, write: Not readable\n"
    "- Do not identify unlabelled pills from appearance.\n"
    "- If no readable label is visible, write: No readable label found — please fill in the fields manually.\n"
    "- Do not diagnose, prescribe, or recommend dose changes.\n\n"
    "Output format:\n\n"
    "---\n"
    "📋 **Extracted from photo — type each value into the matching field below:**\n\n"
    "**Medicine name and strength:** \n"
    "**Amount each time:** \n"
    "**How often:** \n"
    "**Other label instructions:** \n"
    "**Start date and duration:** \n"
    "**Quantity supplied and expiry:** \n\n"
    "---\n"
    "⚠️ Always compare with your actual label before typing values in."
)


def build_content(body):
    file_data = body.get("file_data")
    file_mime = body.get("file_mime", "image/jpeg")
    content   = []
    if file_data:
        if file_mime.startswith("image/"):
            content.append({"type": "image", "source": {"type": "base64", "media_type": file_mime, "data": file_data}})
        else:
            content.append({"type": "document", "source": {"type": "base64", "media_type": file_mime, "data": file_data}})
    content.append({"type": "text", "text": "Please extract the medicine label details."})
    return content


@flask_app.route("/", methods=["OPTIONS"])
def options():
    return Response("", status=200, headers=CORS_HEADERS)


@flask_app.route("/", methods=["POST"])
def handler():
    body     = request.get_json(force=True, silent=True) or {}
    content  = build_content(body)
    messages = [{"role": "user", "content": content}]

    resp = bedrock.invoke_model(
        modelId=MODEL,
        contentType="application/json",
        accept="application/json",
        body=json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1024,
            "system": SYSTEM_PROMPT,
            "messages": messages,
        }),
    )
    result = json.loads(resp["body"].read())
    text   = result.get("content", [{}])[0].get("text", "")
    return Response(text, status=200, content_type="text/plain; charset=utf-8", headers=CORS_HEADERS)


app = Mangum(flask_app)
