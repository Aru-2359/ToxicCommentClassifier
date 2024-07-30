# Toxic Comment Classifier using Bidirectional LSTM🤢🤢

This project implements a toxic comment classifier using a Bidirectional LSTM neural network model. The model is trained to classify comments into multiple toxicity categories.

##Demo
![Demo Screenshot]("demo.png")

## Files

- `Toxicity.ipynb`: Jupyter notebook containing the code for data preprocessing, model definition, training, evaluation, and prediction using the toxic comment classifier.
- `toxicity.h5`: Trained model file saved in HDF5 format, containing the weights and architecture of the Bidirectional LSTM neural network.

## Model Architecture

The neural network architecture is defined using Keras Sequential API with the following layers:

1. **Embedding Layer:**
   - Converts text input into dense vectors of fixed size (`embedding_dim = 32`) and learns embeddings specific to each word. `MAX_FEATURES+1` specifies the maximum number of unique words in the vocabulary plus one for unknown words.

2. **Bidirectional LSTM Layer:**
   - Utilizes Bidirectional Long Short-Term Memory (LSTM) units with `units = 32` and `activation = 'tanh'`. LSTM layers are capable of learning long-term dependencies in sequential data, and bidirectional LSTMs process the input sequence both forwards and backwards.

3. **Dense Layers:**
   - **Dense Layer 1:** 128 neurons with ReLU activation function.
   - **Dense Layer 2:** 256 neurons with ReLU activation function.
   - **Dense Layer 3:** 128 neurons with ReLU activation function.
   - **Output Layer:** 6 neurons with sigmoid activation function for multi-label classification (each neuron corresponds to one toxicity category).

## Usage

1. **Clone the repository:**

   ```bash
   git clone https://github.com/Aru-2359/ToxicCommentClassifier.git
   cd ToxicCommentClassifier
   ```

2. **Open and run the Jupyter notebook:**

   - Ensure you have Jupyter Notebook installed and activated in your environment:

     ```bash
     pip install notebook
     jupyter notebook
     ```

   - Navigate to `Toxicity.ipynb` and open it using Jupyter Notebook interface.

3. **Run the notebook:**
   
   - Execute each cell in the notebook sequentially to preprocess data, define the model, train the model, evaluate performance, and make predictions.
   - Ensure that `toxicity.h5` is in the same directory as `Toxicity.ipynb` or update the notebook to load the model from the correct path.

4. **Adjust as needed:**

   - Modify hyperparameters, model architecture, or dataset handling as necessary for further experimentation or application to different datasets.

## Dependencies

Ensure the following libraries are installed in your Python environment:

- Python 3.x
- TensorFlow 2.x (or compatible version)
- Keras
- NumPy
- Pandas

Install dependencies using:

```bash
pip install tensorflow numpy pandas
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
