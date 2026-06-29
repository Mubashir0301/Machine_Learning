# LSTM Next Word Predictor 🧠📖

Welcome to the **Next Word Predictor** project! This is an interactive deep learning project designed to help you understand how Recurrent Neural Networks, specifically **Long Short-Term Memory (LSTM)** networks, are used to process text sequences and generate language.

---

## 📁 Project Structure

*   **[`next_word_predictor.ipynb`](file:///d:/Machine_Learning/next_word_predictor/next_word_predictor.ipynb)**: A highly documented Python Jupyter Notebook designed for **Google Colab**. It walks through the training pipeline step-by-step using Keras and TensorFlow.
*   **[`index.html`](file:///d:/Machine_Learning/next_word_predictor/index.html)**: The frontend layout of our interactive browser simulator.
*   **[`style.css`](file:///d:/Machine_Learning/next_word_predictor/style.css)**: A premium glassmorphic dark-theme stylesheet.
*   **[`app.js`](file:///d:/Machine_Learning/next_word_predictor/app.js)**: The core logic running **TensorFlow.js** to train and run the LSTM network directly inside your web browser using WebGL GPU acceleration.

---

## 🚀 How to Run the Project

### 1. The Jupyter Notebook (Google Colab)
Since local Windows environments running Python 3.14+ do not support TensorFlow natively yet, Google Colab is the easiest place to run Python deep learning code.
1.  Open your browser and navigate to [Google Colab](https://colab.research.google.com/).
2.  Click **Upload** and select the [`next_word_predictor.ipynb`](file:///d:/Machine_Learning/next_word_predictor/next_word_predictor.ipynb) file.
3.  Run the cells sequentially (Ctrl+Enter or click the play button next to each cell).
4.  Read the explanations in the cells to understand how data is prepared, embeddings are generated, and LSTM gates process sequences.

### 2. The Web Simulator (Interactive browser UI)
The web application runs entirely client-side. **You do not need to install anything or run a command line server!**
1.  Locate the [`index.html`](file:///d:/Machine_Learning/next_word_predictor/index.html) file on your computer.
2.  Double-click it to open it in any web browser (Google Chrome, Microsoft Edge, or Firefox).
3.  Click **Load Sample Text** (or paste your own paragraphs, or upload a `.txt` file).
4.  Configure the training settings on the left sidebar:
    *   **Sequence Length**: The number of words the model looks back to predict the next word.
    *   **Epochs**: Number of training passes. For small sample text, `40` to `60` epochs works best.
5.  Click **Train LSTM Model**. Watch the console log training stages and a chart plot the loss/accuracy in real-time!
6.  Once training is complete, the **Prediction Playground** at the bottom will unlock.
7.  Type a seed word (e.g. `deep learning` or `neural network`) and click **Predict** to watch the LSTM generate subsequent text word-by-word, and review the top 3 predicted word probabilities from the Softmax layer!

---

## 🎓 Key Deep Learning Concepts Explained

Here are the concepts you will learn in this project:

1.  **Tokenization**: The process of mapping words to integer numbers (e.g., `"deep"` = `1`, `"learning"` = `2`). Neurons calculate numbers, not strings.
2.  **Embedding Layer**: Maps those integers to high-dimensional vectors (e.g., size 32). This allows words with similar meanings (like `"machine"` and `"computer"`) to have similar mathematical coordinates.
3.  **LSTM Layer (Long Short-Term Memory)**: Standard feedforward neural networks have no memory. LSTMs contain internal "cells" and "gates" (Forget, Input, and Output gates) that allow them to remember context from words that appeared earlier in the sentence.
4.  **Dense Layer + Softmax**: The final neural layer has one neuron for every word in the vocabulary. The **Softmax** function converts the outputs of these neurons into percentages (probabilities) that sum to 100%. The word with the highest percentage is predicted next.
5.  **Categorical Cross-Entropy Loss**: A loss function that measures how "incorrect" the model's predictions are compared to the actual target next words. The goal of training is to minimize this loss.
