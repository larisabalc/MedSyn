import pandas as pd

class DatasetBuilder:
    """
    Combines mapped Kaggle profiles and synthetic profiles into a final dataset
    suitable for model training. Generates uniform input_text.
    """

    def __init__(self, mapper, factory):
        """
        :param mapper: instance of ProfileMapper
        :param factory: instance of PatientProfileFactory
        """
        self.mapper = mapper
        self.factory = factory

    @staticmethod
    def build_input_text(df):
        """Builds uniform input_text for each row using patient profile + symptoms."""
        def make_text(row):
            ctx = f"The patient is a {row['Age']}-year-old {row['Gender'].lower()}."
            bp = str(row.get("Blood Pressure", "")).lower()
            if "high" in bp:
                ctx += " The patient has high blood pressure."
            elif "low" in bp:
                ctx += " The patient has low blood pressure."
            else:
                ctx += f" The patient has normal blood pressure."

            chol = str(row.get("Cholesterol Level", "")).lower()
            if "high" in chol:
                ctx += " The patient has high cholesterol."
            elif "low" in chol:
                ctx += " The patient has low cholesterol."
            else:
                ctx += " The patient has normal cholesterol."

            symp = row.get("Symptoms", "").strip().lower()
            return ctx + " Reported symptoms include " + symp + "."

        df = df.copy()
        df["input_text"] = df.apply(make_text, axis=1)
        df["target"] = df["Disease"]
        return df[["input_text", "target"]]

    def build(self, n_synthetic_versions=5):
        """
        Combines mapped Kaggle profiles and synthetic profiles into a final dataset.
        :param n_synthetic_versions: Number of synthetic profiles per disease
        """
        mapped_df = self.mapper.map_profiles()

        if "Disease" in mapped_df.columns:
            mapped_df = mapped_df.drop(columns=["Disease"])

        mapped_df = mapped_df.rename(columns={"matched_disease": "Disease"})

        mapped_df = mapped_df[["Disease", "Symptoms", "Age", "Gender", "Blood Pressure", "Cholesterol Level"]]

        synthetic_list = []
        for _, row in self.mapper.symptom_df.iterrows():
            disease_name = row["Name"]
            symptoms = row.get("Symptoms", "")

            self.factory.n_versions = n_synthetic_versions
            synthetic_df = self.factory.generate_multiple_profiles(disease_name, symptoms)

            synthetic_df = synthetic_df[["Disease", "Symptoms", "Age", "Gender", "Blood Pressure", "Cholesterol Level"]]
            synthetic_list.append(synthetic_df)

        synthetic_df = pd.concat(synthetic_list, ignore_index=True)

        final_df = pd.concat([mapped_df, synthetic_df], ignore_index=True)

        final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)

        return self.build_input_text(final_df)



