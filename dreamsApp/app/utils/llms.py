import os
import json
import google.generativeai as genai
from dotenv import load_dotenv
from flask import jsonify
from flask  import current_app
import re


load_dotenv()


def generate(user_id, positive_keywords, negative_keywords):
    client = genai.Client(
        api_key=os.environ.get("GEMINI_API_KEY"),
    )

    input_prompt = f"""
You are a qualitative sociologist analyzing data from a recovery project in Alaska, USA. I have two clusters of keywords based on when a participant felt positive or negative emotions.

Positive Keywords: {positive_keywords}
Negative Keywords: {negative_keywords}

Your task:
- For each cluster, identify 2-3 overarching themes. For each theme, explain how the participant is likely experiencing it emotionally or personally.
- Keep the explanation warm and empathetic, as if you're describing their story gently to a support team.
- Return the output as **valid JSON** in this format:

{{
  "positive": [
    {{
      "theme": "Theme name",
      "meaning": "One sentence explanation"
    }},
    ...
  ],
  "negative": [
    {{
      "theme": "Theme name",
      "meaning": "One sentence explanation"
    }}
  ]
}}
"""

    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=input_prompt)],
        )
    ]

    generate_content_config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_budget=-1),
        tools=[types.Tool(googleSearch=types.GoogleSearch())],
    )

    full_response = ""
    for chunk in client.models.generate_content_stream(
        model="gemini-2.5-flash",
        contents=contents,
        config=generate_content_config,
    ):
        full_response += chunk.text
    match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', full_response, re.DOTALL)
    if match:
        cleaned_response = match.group(1)
    else:
        cleaned_response = full_response 

    try:
        data = json.loads(cleaned_response)
        mongo = current_app.mongo
        
        result = mongo['thematic_analysis'].update_one(
            {"user_id": str(user_id)},
            {"$set": {"data": data}},
            upsert=True
        )

        return jsonify(data)

    except json.JSONDecodeError:
        return jsonify({"error": "Invalid JSON from Gemini", "raw": full_response})
    except Exception as e:
        return jsonify({"error": "Unexpected error", "details": str(e)})


