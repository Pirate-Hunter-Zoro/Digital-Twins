import os
from typing import List
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
import hashlib
import torch
import json
from scripts.patient_embedding.shared.similarity import cosine

from dotenv import load_dotenv
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
            device = os.environ['EMBEDDER_DEVICE'] if torch.cuda.is_available() else 'cpu',
            trust_remote_code=True
        )
        self.vectors_path = Path(os.environ['VECTORS_DIR'])
        os.makedirs(self.vectors_path, exist_ok=True)
        
        # Whether vectors are to be scrubbed and recomputing
        self.scrub_vectors = int(os.environ['SCRUB_VECTORS']) == 1
        
    def _generate_id(self, text: str) -> str:
        """
        Generate unique ID pertaining to the input string
        
        :param text: String to give ID to
        :type text: str
        :return: Resulting ID
        :rtype: str
        """
        return hashlib.md5(text.encode('utf-8')).hexdigest()

    def vectorize(self, strings: list[str], instruction: str=None) -> List[np.array]:
        """
        Generates normalized vector embeddings for a batch of texts using a simple .encode() call.
        
        :param strings: List of strings to embed
        :type strings: list[str]
        :param instruction: Specific instructions to provide the embedder when it vectorizes
        :type instruction: str
        :return: Resulting vectors
        :rtype: List
        """
        # The encode method handles tokenization, inference, and pooling.
        vectors = [None for _ in strings]
        to_compute = []
        to_compute_indices = []
        for i, string in enumerate(strings):
            id = self._generate_id(text=string)
            vectors_store_path = self.vectors_path / f"vector_{id}.npy"
            if vectors_store_path.exists() and not self.scrub_vectors:
                vectors[i] = np.load(vectors_store_path)
            else:
                to_compute.append(string)
                to_compute_indices.append(i)
                
        if instruction is not None:
            model_input = [f"Instruct: {instruction}\nQuery: {text}" for text in to_compute]
        else:
            model_input = to_compute
            
        if len(to_compute) > 0:
            missing_vectors = self.model.encode(
                model_input,
                normalize_embeddings=True,
                show_progress_bar=True,
                convert_to_numpy=True,
                batch_size=int(os.environ['EMBEDDER_BATCH_SIZE'])
            )
            for i, missing_vector in zip(to_compute_indices, missing_vectors):
                vectors[i] = missing_vector
                id = self._generate_id(text=strings[i])
                vectors_store_path = self.vectors_path / f"vector_{id}.npy"
                np.save(vectors_store_path, missing_vector)
                string_store_path = self.vectors_path / f"string_{id}.txt"
                with open(string_store_path, 'w') as f:
                    f.write(strings[i])
                
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
    vecs = embedder.vectorize([str_1, str_2], instruction="Retrieve medical cases")