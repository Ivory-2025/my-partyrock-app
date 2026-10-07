"""
Lambda: my_return_plan
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
    "You are a healthcare directory assistant helping a patient return unused medicines. "
    "Be concise and scannable. Use emojis.\n\n"
    "RULES:\n"
    "- Accept ALL medicine types for return.\n"
    "- If both Return Item Details and photo are empty: 💊 Please enter a medicine name or upload a labelled photo.\n"
    "- Do not identify unlabelled pills from appearance.\n"
    "- Do not recommend flushing, household disposal, donation, or reuse.\n"
    "- If location is vague: label UNCONFIRMED LOCATION and warn to verify via myMediSAFE.\n\n"
    "Generate:\n\n"
    "## ♻️ My Return Summary\n"
    "💊 Medicine | 🏷️ Type | 🔢 Quantity | ❓ Reason\n\n"
    "📦 How to prepare — 4 bullet points\n\n"
    "📍 Nearby collection points — up to 4 facilities with Google Maps links\n\n"
    "🔍 General search links:\n"
    "- MyMediSAFE: https://www.mymedisafe.org.my/\n"
    "- Pharmacies: https://www.google.com/maps/search/pharmacy+near+[LOCATION]\n\n"
    "❓ Questions to confirm before going — 2 to 3 bullets\n\n"
    "*🚫 Do not flush or bin medicines. Call ahead to confirm acceptance.*"
)


@flask_app.route("/", methods=["OPTIONS"])
def options():
    return Response("", status=200, headers=CORS_HEADERS)


@flask_app.route("/", methods=["POST"])
def handler():
    body      = request.get_json(force=True, silent=True) or {}
    lang      = body.get("preferred_language", "English")
    item      = body.get("return_item_details", "")
    location  = body.get("location", "")
    file_data = body.get("file_data")
    file_mime = body.get("file_mime", "image/jpeg")

    content = []
    if file_data:
        if file_mime.startswith("image/"):
            content.append({"type": "image", "source": {"type": "base64", "media_type": file_mime, "data": file_data}})
    content.append({"type": "text", "text": f"Respond in {lang}.\nReturn item: {item or '(none)'}\nLocation: {location or '(none)'}\nGenerate the Return Plan."})

    resp = bedrock.invoke_model(
        modelId=MODEL,
        contentType="application/json",
        accept="application/json",
        body=json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1500,
            "system": SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": content}],
        }),
    )
    result = json.loads(resp["body"].read())
    text   = result.get("content", [{}])[0].get("text", "")
    return Response(text, status=200, content_type="text/plain; charset=utf-8", headers=CORS_HEADERS)


app = Mangum(flask_app)
