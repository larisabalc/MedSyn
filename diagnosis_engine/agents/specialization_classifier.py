from mistralai import Mistral
import json
import re

class SpecializationClassifier:
    def __init__(self, api_key: str):
        self.client = Mistral(api_key=api_key)

    def classify(self, diagnosis: str):
        prompt = f"""
        You are a medical classification assistant.

        Given a diagnosis, determine the most appropriate medical specialization.

        Diagnosis:
        {diagnosis}

        Return JSON only:
        {{
            "specialization": "..."
        }}
        """

        result = self.client.chat.complete(
            model="mistral-small-latest",
            messages=[{"role": "user", "content": prompt}]
        )

        json_raw = result.choices[0].message.content
        data = self.extract_json_from_text(json_raw)
        specialization = data.get("specialization", "").strip()

        return specialization if specialization else "General Practitioner"

    @staticmethod
    def extract_json_from_text(text: str):
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return {}
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            return {}