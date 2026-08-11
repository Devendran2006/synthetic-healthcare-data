import os
import pandas as pd
from sdv.metadata import SingleTableMetadata
from sdv.single_table import CTGANSynthesizer

# Load CTGAN-ready dataset
df = pd.read_csv("data/ctgan_ready.csv")

print("Dataset Shape:", df.shape)

# Detect metadata
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(df)

# Create CTGAN model
synthesizer = CTGANSynthesizer(
    metadata=metadata,
    epochs=150,
    verbose=True
)

print("Training CTGAN Model...")
synthesizer.fit(df)

print("Generating Synthetic Data...")

synthetic_data = synthesizer.sample(num_rows=55500)


os.makedirs("outputs", exist_ok=True)

synthetic_data.to_csv(
    "outputs/synthetic_healthcare.csv",
    index=False
)
import joblib

joblib.dump(
    synthesizer,
    "models/ctgan_model.pkl"
)
print("Synthetic Dataset Generated Successfully!")
print(synthetic_data.head())
print("Shape:", synthetic_data.shape)