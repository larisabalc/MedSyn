import pandas as pd
import re
from rapidfuzz import process, fuzz
from datasets import load_dataset

class ProfileMapper:
    """
    Maps positive Kaggle patient profiles to the symptom-disease dataset
    using RapidFuzz distance-based matching. Does NOT create final sentences.
    """

    def __init__(self, kaggle_csv_path, symptom_dataset=None, threshold=65):
        self.df_kaggle = pd.read_csv(kaggle_csv_path)
        
        self.df_kaggle = self.df_kaggle[self.df_kaggle["Outcome Variable"].str.lower() == "positive"]
        self.threshold = threshold

        if symptom_dataset is None:
            ds = load_dataset("QuyenAnhDE/Diseases_Symptoms")
            self.symptom_df = ds["train"].to_pandas()
        else:
            self.symptom_df = symptom_dataset

        self.df_kaggle["disease_norm"] = self.df_kaggle["Disease"].apply(self.normalize_name)
        self.symptom_df["disease_norm"] = self.symptom_df["Name"].apply(self.normalize_name)

        self.sym_norm_list = self.symptom_df["disease_norm"].unique().tolist()
        self.norm_to_name = dict(zip(self.symptom_df["disease_norm"], self.symptom_df["Name"]))

    @staticmethod
    def normalize_name(name):
        if pd.isna(name):
            return ""
        s = name.strip().lower()
        s = re.sub(r"[^\w\s]", " ", s)
        s = re.sub(r"\s+", " ", s)
        return s.strip()

    def find_closest_disease(self, norm_name):
        match = process.extractOne(norm_name, self.sym_norm_list, scorer=fuzz.ratio)
        if match:
            matched_name, score, _ = match
            if score >= self.threshold:
                return matched_name
        return None

    def map_profiles(self):
        """
        Returns a DataFrame with:
        - original profile columns
        - matched disease name from symptom dataset
        - corresponding symptom list
        Does NOT generate input_text.
        """
        self.df_kaggle["matched_norm"] = self.df_kaggle["disease_norm"].apply(self.find_closest_disease)
        self.df_kaggle["matched_disease"] = self.df_kaggle["matched_norm"].map(self.norm_to_name)

        matched = self.df_kaggle.dropna(subset=["matched_disease"]).copy()

        merged = matched.merge(self.symptom_df, left_on="matched_disease", right_on="Name", how="inner")

        return merged

    def get_unmatched_diseases(self):
        hf_set = set(self.symptom_df["disease_norm"])                   
        matched_hf_set = set(self.df_kaggle["matched_norm"].dropna()) 
        return hf_set - matched_hf_set  
