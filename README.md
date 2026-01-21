# Sentiment-Analysis
#  Emotion Classification using Bidirectional LSTM

This project is a Deep Learning model built with **TensorFlow** and **Keras** to classify text (tweets) into six distinct emotional categories. It uses a Recurrent Neural Network (RNN) architecture to understand the context and sentiment of human language.

##  Dataset
The model is trained on the **dair-ai/emotion** dataset from Hugging Face.


## Model Architecture
The model utilizes a sophisticated NLP pipeline to process and understand text:
1.  **Tokenization:** Converts words into numerical tokens using a 10,000-word vocabulary.
2.  **Embedding Layer:** Maps words into a 16-dimensional vector space to capture semantic relationships.
3.  **Bidirectional LSTMs:** Two stacked layers of Long Short-Term Memory units that read the text both forwards and backwards to capture deep contextual meaning.
4.  **Dense Output Layer:** A 6-unit layer with a Softmax activation to provide probability scores for each emotion class.



##  Visualizations
The script provides deep insights into the model's performance through:
* **Training History:** Side-by-side plots of Accuracy and Loss for both training and validation sets to monitor learning and identify overfitting.
* **Confusion Matrix:** A normalized heat map that visualizes the model's precision across all six emotional categories.
* **Random Sampling:** A testing module that picks random tweets and compares the ground truth emotion against the model's prediction.



##  Built With
* [TensorFlow](https://www.tensorflow.org/) - Deep Learning Framework
* [Hugging Face Datasets](https://huggingface.co/docs/datasets/index) - Data Loading
* [Matplotlib](https://matplotlib.org/) - Data Visualization
* [Scikit-learn](https://scikit-learn.org/) - Metrics and Evaluation
