"""
Lambda: my_medicine_summary
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
    "You are a medicine-information assistant helping a patient prepare for a pharmacist visit. "
    "Be concise and friendly. Use emojis to make sections easy to scan.\n\n"
    "RULES:\n"
    "- Treat all inputs as patient-reported or AI-extracted draft.\n"
    "- If Medicine Card is empty: No medicine details found. Complete Section 1 first.\n"
    "- Blank Other Medicines = Not provided. Blank Allergies = Not provided.\n"
    "- Never say safe, no interactions, or imply medical clearance.\n\n"
    "Generate these sections:\n\n"
    "💊 Your Instructions — name, strength, dose, frequency, key directions.\n\n"
    "👀 Watch Out For — up to 4 bullets of what to watch for.\n\n"
    "🍽️ Food and Drink — confirmed interactions only. If none: confirm with pharmacist.\n\n"
    "🗣️ Concerns to Discuss — one bullet per concern.\n\n"
    "❓ Ask Your Pharmacist — 3 specific questions as if the patient is speaking.\n\n"
    "📚 Sources:\n"
    "- Drugs.com — https://www.drugs.com/drug_interactions.html\n"
    "- MedlinePlus — https://medlineplus.gov/druginformation.html\n"
    "- Malaysian DCA — https://www.pharmacy.gov.my\n\n"
    "*⚠️ AI-extracted and patient-reported info. Does not replace pharmacist advice.*"
)


@flask_app.route("/", methods=["OPTIONS"])
def options():
    return Response("", status=200, headers=CORS_HEADERS)


@flask_app.route("/", methods=["POST"])
def handler():
    body    = request.get_json(force=True, silent=True) or {}
    lang    = body.get("preferred_language", "English")
    card    = body.get("medicine_card", "")
    age     = body.get("patient_age", "Not provided")
    others  = body.get("other_medicines", "Not provided")
    allergy = body.get("allergies", "Not provided")

    user_text = (
        f"Respond in {lang}.\n\n"
        f"Medicine Card:\n{card or '(none)'}\n\n"
        f"Age group: {age}\nOther medicines: {others}\nAllergies: {allergy}\n\n"
        "Generate the Medicine Summary."
    )

    resp = bedrock.invoke_model(
        modelId=MODEL,
        contentType="application/json",
        accept="application/json",
        body=json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1500,
            "system": SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": user_text}],
        }),
    )
    result = json.loads(resp["body"].read())
    text   = result.get("content", [{}])[0].get("text", "")
    return Response(text, status=200, content_type="text/plain; charset=utf-8", headers=CORS_HEADERS)


app = Mangum(flask_app)
