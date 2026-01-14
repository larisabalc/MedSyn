from diagnosis_engine.diagnosis_service import DiagnosisService
from diagnosis_engine.models.context_diagnosis_classifier import ContextDiagnosisClassifier
from ocr_service.ocr_engine import OCREngine
from ocr_service.medical_extractor import MedicalInfoExtractor

from dotenv import load_dotenv
import os
import json

load_dotenv()
API_KEY = os.getenv("MISTRAL_API_KEY")

def main():
    ocr = OCREngine(API_KEY)
    extractor = MedicalInfoExtractor(API_KEY)
    file_path = 'patient-medical-record-template_x.png'

    text = ocr.extract_text(file_path)
    info = extractor.extract(text)

    context_strategy = ContextDiagnosisClassifier()
    context_strategy.load_model("diagnosis_engine/trained_models/context")

    service = DiagnosisService(strategy=context_strategy)

    patient_input = (
    f"The patient is a {info['age']}-year-old patient. "
    f"Reported symptoms include {', '.join(info['symptoms'])}."
)
    print(patient_input)

    diagnosis = service.predict(patient_input)
    print("Predicted diagnosis:", diagnosis)

if __name__ == "__main__":
    main()
