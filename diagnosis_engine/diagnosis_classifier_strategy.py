from diagnosis_engine.prediction_strategy import PredictionStrategy
from diagnosis_engine.models.context_diagnosis_classifier import ContextDiagnosisClassifier

class DiagnosisClassifierStrategy(PredictionStrategy):
    def __init__(self):
        self.model = ContextDiagnosisClassifier()

    def load_model(self, model_path):
        self.model.load_model(model_path)

    def generate_disease_name(self, symptom_description):
        return self.model.generate_disease_name(symptom_description)