# Next Word Prediction using LSTM

## 📌 Project Overview
This project implements a **Next Word Prediction model** using **Long Short-Term Memory (LSTM)** networks.  
The model is trained on textual data to predict the next word in a sequence, a common **Natural Language Processing (NLP)** task.

LSTMs are a type of Recurrent Neural Network (RNN) that can capture long-term dependencies, making them suitable for text-based sequence prediction.

---

## 🚀 Features
- Data preprocessing (tokenization, padding, sequence preparation)
- LSTM-based deep learning model
- Training and evaluation of the model
- Prediction of the next word given an input sequence
- Extendable to larger text corpora for improved accuracy

---

## 📂 Project Structure
- `next word prediction.ipynb` → Jupyter Notebook containing the full implementation  
- `README.md` → Project description and usage instructions  

---

## 🛠️ Technologies Used
- Python  
- TensorFlow / Keras  
- NumPy, Pandas  
- NLTK / Keras Tokenizer (for text preprocessing)  
- Matplotlib (for visualization)

---

## 📊 Workflow
1. **Data Collection** – Input text corpus used for training.  
2. **Data Preprocessing** – Tokenization, vocabulary creation, sequence preparation.  
3. **Model Building** – LSTM-based deep learning model.  
4. **Training** – Model trained on prepared sequences.  
5. **Evaluation** – Assess accuracy and performance.  
6. **Prediction** – Generate the most likely next word for a given input sequence.  

---

## 📈 Example Usage
```python
input_text = "Artificial Intelligence is"
predicted_word = model.predict(input_text)
print(predicted_word)  # e.g., "future"
