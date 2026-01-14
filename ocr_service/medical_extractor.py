from mistralai import Mistral
import json
import re

class MedicalInfoExtractor:
    def __init__(self, api_key: str):
        self.client = Mistral(api_key=api_key)

    def extract(self, text: str):
        prompt = f"""
        You are an information extraction assistant.
        Extract the following fields ONLY from the text below:

        - Age (if DOB is present, calculate)
        - Gender (Male/Female/Unknown)
        - Symptoms (list of symptom phrases only)
        - Medical conditions (if listed)
        - Allergies
        - Medications

        Text:
        {text}

        Return JSON:
        {{
        "age": ...,
        "gender": "...",
        "symptoms": [...],
        "conditions": [...],
        "allergies": [...],
        "medications": [...]
        }}
        """

        result = self.client.chat.complete(
            model="mistral-large-latest",
            messages=[{"role": "user", "content": prompt}]
        )

        json_raw = result.choices[0].message.content
        json_clean = re.sub(r"^```json\s*|\s*```$", "", json_raw.strip(), flags=re.MULTILINE)
        info = json.loads(json_clean)
        return info
