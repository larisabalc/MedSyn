import base64
from mistralai import Mistral

class OCREngine:
    def __init__(self, api_key: str):
        self.client = Mistral(api_key=api_key)

    def extract_text(self, file_path: str):
        with open(file_path, "rb") as f:
            file_bytes = f.read()
        file_b64 = base64.b64encode(file_bytes).decode("utf-8")

        response = self.client.ocr.process(
            model="mistral-ocr-latest",
            document={
                "type": "document_url",
                "document_url": (
                    f"data:application/pdf;base64,{file_b64}"
                )
            },
            include_image_base64=False
        )

        pages_text = "\n\n".join(p.markdown for p in response.pages)
        return pages_text
