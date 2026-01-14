from diagnosis_engine.prediction_strategy import PredictionStrategy

class DiagnosisService:
    def __init__(self, strategy: PredictionStrategy):
        self.strategy = strategy

    def set_strategy(self, strategy: PredictionStrategy):
        self.strategy = strategy

    def train(self, *args, **kwargs):
        self.strategy.train(*args, **kwargs)

    def evaluate(self, *args, **kwargs):
        return self.strategy.evaluate(*args, **kwargs)

    def predict(self, patient_description):
        return self.strategy.generate_disease_name(patient_description)

    def save_model(self, path):
        self.strategy.save_model(path)

    def load_model(self, path):
        self.strategy.load_model(path)
