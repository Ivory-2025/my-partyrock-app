"""
Lambda: extract_from_photo
Reads a medicine label photo and extracts fields for the user to type in.
"""
import json, os, boto3
from flask import Flask, request, Response, stream_with_context

app = Flask(__name__)
bedrock = boto3.client("bedrock-runtime", region_name=os.environ.get("AWS_REGION", "ap-southeast-1"))
MODEL   = "global.anthropic.claude-haiku-4-5-20251001-v1:0"

CORS = {
    "Access-Control-Allow-Origin":  "*",
    "Access-Control-Allow-Headers": "Content-Type",
    "Access-Control-Allow-Methods": "POST,OPTIONS",
}

SYSTEM_PROMPT = (
    "You are a clinical pharmacist assistant. A patient has uploaded a photo or scan "
    "of a medicine label. Extract the details and display them field by field so the "
    "patient can type each value into the matching input box below.\n\n"
    "RULES:\n"
    "- Extract only what is explicitly and clearly visible on the label. Do not infer or calculate missing fields.\n"
    "- If a field is not visible or not readable, write: Not readable\n"
    "- Do not identify unlabelled pills from appearance. If no readable label is visible, write: "
    "No readable label found — please fill in the fields manually.\n"
    "- Do not diagnose, prescribe, or recommend dose changes.\n"
    "- If no photo is uploaded, write: No photo uploaded yet. Upload a photo on the left to extract label details.\n\n"
    "Output the extracted details in this exact format:\n\n"
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


def build_messages(body: dict) -> list:
    file_data = body.get("file_data")
    file_mime = body.get("file_mime", "image/jpeg")
    content   = []

    if file_data:
        if file_mime.startswith("image/"):
            content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": file_mime, "data": file_data},
            })
        else:
            content.append({
                "type": "document",
                "source": {"type": "base64", "media_type": file_mime, "data": file_data},
            })

    content.append({"type": "text", "text": "Please extract the medicine label details."})
    return [{"role": "user", "content": content}]


@app.route("/", methods=["OPTIONS"])
def options():
    return Response("", status=200, headers=CORS)


@app.route("/", methods=["POST"])
def handler():
    body = request.get_json(force=True, silent=True) or {}

    messages = build_messages(body)

    def generate():
        resp = bedrock.invoke_model_with_response_stream(
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
        for event in resp["body"]:
            chunk = event.get("chunk")
            if chunk:
                data = json.loads(chunk["bytes"].decode())
                if data.get("type") == "content_block_delta":
                    text = data.get("delta", {}).get("text", "")
                    if text:
                        yield text

    headers = {**CORS, "X-Accel-Buffering": "no", "Cache-Control": "no-cache"}
    return Response(stream_with_context(generate()),
                    content_type="text/plain; charset=utf-8",
                    headers=headers)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
