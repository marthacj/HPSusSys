from flask import Flask, request, jsonify
from transformers import pipeline

app = Flask(__name__)

# Load your model and tokenizer
qa_pipeline = pipeline("question-answering", model="path/to/your/model", tokenizer="path/to/your/tokenizer")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    question = data.get('question')
    context = data.get('context')
    result = qa_pipeline(question=question, context=context)
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

# from flask import Flask
# app = Flask(__name__)

# @app.route("/")
# def hello():
#     return "Hello World!"

# if __name__ == "__main__":
#     app.run(port=5001)

# from transformers import AutoTokenizer, AutoModelForQuestionAnswering
# from langchain import LanguageChain, LocalDocumentStore, LocalModel
# import pandas as pd

# def preprocess_telemetry_data(file_path):
#     telemetry_df = pd.read_csv(file_path)
#     telemetry_text = ""
#     for index, row in telemetry_df.iterrows():
#         telemetry_text += f"At time {row['timestamp']}, the value of {row['metric_name']} was {row['value']}.\n"
#     return telemetry_text

# def main():
#     # Preprocess telemetry data
#     telemetry_text = preprocess_telemetry_data('./data/telemetry_data.csv')

#     # Load the model and tokenizer from local directory
#     tokenizer = AutoTokenizer.from_pretrained("./models/local_model_directory")
#     model = AutoModelForQuestionAnswering.from_pretrained("./models/local_model_directory")

#     # Initialize LangChain components
#     document_store = LocalDocumentStore(directory="./data")
#     lang_chain_model = LocalModel(directory="./models/local_model_directory")

#     # Create a LangChain application
#     lang_chain = LanguageChain(model=lang_chain_model, document_store=document_store)

#     # Example question answering
#     qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)
#     question = "What was the value of temperature at 10:00 AM?"
#     result = qa_pipeline({'question': question, 'context': telemetry_text})
    
#     print(f"Answer: {result['answer']}")

# if __name__ == "__main__":
#     main()