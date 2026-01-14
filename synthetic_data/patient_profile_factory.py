from synthetic_data.heuristics import assign_gender, assign_age, assign_blood_pressure, assign_cholesterol
import pandas as pd

class PatientProfileFactory:
    """
    Generates synthetic patient profiles for a given disease + symptoms.
    """

    def __init__(self, n_versions=5):
        self.n_versions = n_versions

    def generate_profile(self, disease_name, symptoms=None):
        return {
            "Disease": disease_name,
            "Symptoms": symptoms,
            "Gender": assign_gender(disease_name, symptoms),
            "Age": assign_age(disease_name, symptoms),
            "Blood Pressure": assign_blood_pressure(disease_name, symptoms),
            "Cholesterol Level": assign_cholesterol(disease_name, symptoms)
        }

    def generate_multiple_profiles(self, disease_name, symptoms=None, as_dataframe=True):
        profiles = [self.generate_profile(disease_name, symptoms) for _ in range(self.n_versions)]
        if as_dataframe:
            return pd.DataFrame(profiles)
        return profiles
