import os
from typing import List
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import torch
from scripts.patient_embedding.shared.similarity import cosine

load_dotenv()

class StringEmbedder:
    """
    A simplified class that uses the sentence-transformers library to embed text.
    It handles all the underlying complexity.
    """
   
    def __init__(self):
        """
        Loads a SentenceTransformer model from a local path.
        The library automatically handles device placement.
        """
        full_model_path = Path(os.environ['EMBEDDER_MODEL_PATH'])

        if not os.path.isdir(full_model_path):
            raise FileNotFoundError(f"Pathetic. Model directory not found: {full_model_path}")

        model_name = os.environ['EMBEDDER_MODEL_NAME']
        print(f"[PatientEmbedder] Loading SentenceTransformer model '{model_name}'.")

        # The library handles everything: loading the model, tokenizer, and pooling configuration.
        self.model = SentenceTransformer(
            model_name_or_path = str(full_model_path),
            device = os.environ['EMBEDDER_DEVICE'] if torch.cuda.is_available() else 'cpu'
        )

    def vectorize(self, narratives: list[str]) -> List[np.array]:
        """
        Generates normalized vector embeddings for a batch of texts using a simple .encode() call.
        """
        # The encode method handles tokenization, inference, and pooling.
        # normalize_embeddings=True is the same as the manual normalization you were doing.
        vectors = self.model.encode(
            narratives,
            normalize_embeddings=True,
            show_progress_bar=True
        )
        
        return [vec.astype(np.float32) for vec in vectors]
    
    
if __name__=="__main__":
    str_1 = "## Diagnostics (labs, radiology, vitals, procedures)\
- Encounter for immunization, -621 days\
- Encounter for general adult medical examination without abnormal findings, -251 days"
    str_2 = "## Diagnostics (labs, radiology, vitals, procedures)\
- Abnormal stress test (R94.39), -1 days\
- Abnormal result of other cardiovascular function study (R94.39), -1 days\
- Body mass index (BMI) 37.0-37.9, adult (Z68.37), -1 days\
- Chest pain (R07.9), -1 days\
- Chest pain, unspecified (R07.9), -1 days\
- Chest pain, unspecified type (R07.9), -1 days\
- Coronary artery disease of native artery of native heart with stable angina pectoris (I25.118), -1 days\
- Coronary artery disease involving native coronary artery of native heart without angina pectoris (I25.10), -1 days\
- Dyslipidemia associated with type 2 diabetes mellitus  (CMS/HCC, HHS/HCC) (E78.5), -1 days\
- Family history of malignant neoplasm of breast (Z80.3), -1 days\
- Family history of malignant neoplasm of digestive organs (Z80.0), -1 days\
- Family history of ischemic heart disease and other diseases of the circulatory system (Z82.49), -1 days\
- Hypokalemia (E87.6), -1 days\
- Hyperlipidemia, unspecified (E78.5), -1 days\
- Hypertension associated with diabetes  (CMS/HCC, HHS/HCC) (I15.2), -1 days\
- Long term (current) use of antithrombotics/antiplatelets (Z79.02), -1 days\
- Long term (current) use of aspirin (Z79.82), -1 days\
- Long term (current) use of oral hypoglycemic drugs (Z79.84), -1 days\
- Major depressive disorder, single episode, mild (F32.0), -1 days\
- Narcolepsy without cataplexy (HHS/HCC) (G47.419), -1 days\
- Obesity, unspecified (E66.9), -1 days\
- Old myocardial infarction (I25.2), -1 days\
- Other forms of angina pectoris (I20.8), -1 days\
- Other long term (current) drug therapy (Z79.899), -1 days\
- Other specified cardiac arrhythmias (I49.8), -1 days\
- Other specified symptoms and signs involving the circulatory and respiratory systems (R09.89), -1 days\
- Other thrombophilia (HHS/HCC) (D68.69), -1 days\
- Personal history of nicotine dependence (Z87.891), -1 days\
- Precordial pain (R07.2), -1 days\
- Type 2 diabetes mellitus with hyperglycemia (CMS/HCC, HHS/HCC) (E11.65), -1 days\
- Type 2 diabetes mellitus with other specified complication (CMS/HCC, HHS/HCC) (E11.69), -1 days\
- Type 2 diabetes mellitus without complications (CMS/HCC, HHS/HCC) (E11.9), -1 days\
- Vitamin D deficiency, unspecified (E55.9), -307 days\
- Other forms of angina pectoris (I20.8), -307 days\
- Anxiety disorder, unspecified (F41.9), -664 days\
- Allergy status to narcotic agent (Z88.5), -664 days\
- Allergy status to sulfonamides (Z88.2), -664 days\
- Atherosclerotic heart disease of native coronary artery without angina pectoris (I25.10), -664 days\
- Chest pain, unspecified (R07.9), -664 days\
- Chest pain, unspecified type (R07.9), -664 days\
- Coronary artery disease involving native heart, angina presence unspecified, unspecified vessel or lesion type (I25.10), -664 days\
- Coronary artery disease of native artery of native heart with stable angina pectoris (I25.118), -664 days\
- Controlled type 2 diabetes mellitus without complication, without long-term current use of insulin (CMS/HCC, HHS/HCC) (E11.9), -664 days\
- Essential (primary) hypertension (I10), -664 days\
- Essential hypertension (I10), -664 days\
- History of non-ST elevation myocardial infarction (NSTEMI) (I25.2), -664 days\
- Hypertension associated with diabetes  (CMS/HCC, HHS/HCC) (E11.59), -664 days\
- Long term (current) use of aspirin (Z79.82), -664 days\
- MI (myocardial infarction)  (CMS/HCC, HHS/HCC) (I21.9), -664 days\
- Old myocardial infarction (I25.2), -664 days\
- Personal history of nicotine dependence (Z87.891), -664 days\
- Precordial pain (R07.2), -664 days\
- Tobacco use (Z72.0), -714 days\
- Non-ST elevation (NSTEMI) myocardial infarction (CMS/HCC, HHS/HCC) (I21.4), -714 days\
- Essential (primary) hypertension (I10), -714 days\
- XR CHEST 1 VIEW, -665 days"

    embedder = StringEmbedder()
    vec1_first = embedder.vectorize([str_1])[0]
    vec1_second = embedder.vectorize([str_1])[0]
    vec2_first = embedder.vectorize([str_2])[0]
    vec2_second = embedder.vectorize([str_2])[0]
    print(cosine(vec1_first, vec2_first))
    print(cosine(vec2_second, vec1_second))
    print(cosine(vec1_first, vec1_second))
    print(cosine(vec2_first, vec2_second))
    
    empty_str_vec = embedder.vectorize([""])[0]
    print(cosine(vec1_first, empty_str_vec))
    print(cosine(empty_str_vec, empty_str_vec))