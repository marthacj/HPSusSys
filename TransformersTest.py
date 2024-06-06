import subprocess
import os

# Manually add the path to ffmpeg to PATH
ffmpeg_path = r'C:\ffmpeg\ffmpeg\bin'  # Update this path to the actual location
os.environ['PATH'] += os.pathsep + ffmpeg_path

# Check if ffmpeg is available
try:
    subprocess.run(["ffmpeg", "-version"], check=True)
    print("ffmpeg is available.")
except FileNotFoundError:
    print("ffmpeg is not found. Please ensure ffmpeg is installed and added to the PATH.")
    exit(1)

from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline


tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")

nlp = pipeline("ner", model=model, tokenizer=tokenizer)
example = "My name is Wolfgang and I live in Berlin"

ner_results = nlp(example)
print(ner_results)

transcriber = pipeline(model="openai/whisper-large-v2")
transcription = transcriber(r"c:\Users\martha.calder-jones\Downloads\71 Altenburg Gardens Management.mp3")
print(transcription)