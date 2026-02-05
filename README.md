# A3: English-to-Nepali Machine Translation with Transformer and Attention Mechanisms

This project implements an English-to-Nepali machine translation system using a Transformer-based sequence-to-sequence model with three different attention mechanisms: General, Multiplicative, and Additive Attention. The implementation includes model training, evaluation, and a web application for real-time translation.

## Features

- **Transformer Architecture**: Custom implementation of Encoder-Decoder Transformer with multi-head attention
- **Attention Mechanisms**: Comparison of three attention types:
  - General Attention
  - Multiplicative Attention
  - Additive Attention
- **Tokenization**: English (spaCy) and Nepali (WordPiece) tokenization
- **Web Application**: React frontend and Django backend for translation interface
- **Model Evaluation**: Training and validation metrics with perplexity scores

## Project Structure

```
A3/
├── A3.ipynb                 # Main Jupyter notebook with implementation
├── A3 copy.ipynb           # Backup copy of the notebook
├── README.md               # This file
├── model/                  # Saved model weights and vocabularies
├── translation-app/        # Web application
│   ├── frontend/          # React application
│   └── backend/           # Django API server
└── requirements.txt       # Python dependencies
```

## Installation

### Prerequisites

- Python 3.8+
- Node.js 16+ (for React frontend)
- Git

### Setup

1. **Clone or navigate to the project directory:**
   ```bash
   cd D:\AIT\Semester II\NLP\Code\A3
   ```

2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Setup the web application:**

   **Backend (Django):**
   ```bash
   cd translation-app/backend
   pip install -r requirements.txt
   python manage.py migrate
   ```

   **Frontend (React):**
   ```bash
   cd translation-app/frontend
   npm install
   ```

## Usage

### Training the Models

1. Open `A3.ipynb` in Jupyter Notebook or VS Code
2. Run the cells in order to:
   - Load and preprocess the dataset
   - Train models with different attention mechanisms
   - Evaluate and compare performance

### Running the Web Application

1. **Start the Django backend:**
   ```bash
   cd translation-app/backend
   python manage.py runserver
   ```

2. **Start the React frontend:**
   ```bash
   cd translation-app/frontend
   npm start
   ```

3. Open your browser to `http://localhost:3000` for the translation interface

### API Usage

The Django backend provides a REST API endpoint:

- **POST /translate/**
  - Input: `{"text": "English sentence to translate"}`
  - Output: `{"translation": "Translated Nepali text"}`

## Model Details

### Architecture

- **Encoder**: 6 layers, 8 attention heads, 512 hidden dimensions
- **Decoder**: 6 layers, 8 attention heads, 512 hidden dimensions
- **Vocabulary**: ~30,000 tokens for both languages
- **Attention Types**:
  - General: Standard dot-product attention
  - Multiplicative: Scaled dot-product attention
  - Additive: Bahdanau-style attention

### Training Parameters

- Batch size: 128
- Learning rate: 0.0005
- Epochs: 5
- Optimizer: Adam
- Loss: Cross-entropy with label smoothing

## Results

### Performance Comparison

| Attention Type         | Training Loss | Training PPL | Validation Loss | Validation PPL |
|------------------------|---------------|--------------|-----------------|---------------|
| General Attention      | 7.083        | 1191.354    | 6.542          | 693.810      |
| Multiplicative Attention| 6.775       | 875.502     | 6.174          | 479.904      |
| Additive Attention     | 6.969        | 1062.796    | 6.368          | 583.169      |

**Best Performing Model**: Multiplicative Attention (lowest validation perplexity)

### Analysis

- All models show improvement on validation data
- Multiplicative Attention performs best overall
- Translation quality is limited due to low training epochs
- Potential for improvement with more data and training time

## Web Application

### Frontend (React)

- Clean, responsive UI for text input and translation display
- Real-time API communication
- Error handling and loading states

### Backend (Django)

- RESTful API for translation requests
- Model loading and inference
- CORS enabled for frontend communication

## Future Improvements

- Increase training epochs for better convergence
- Implement beam search for improved translation quality
- Add more training data
- Fine-tune hyperparameters
- Deploy to cloud platform
- Add support for batch translation

## Dependencies

### Python
- torch>=1.9.0
- transformers
- datasets
- spacy
- nepalitokenizers
- django
- djangorestframework
- django-cors-headers
- numpy
- pandas
- matplotlib

### JavaScript (Frontend)
- react
- axios
- bootstrap (or similar UI library)

## License

This project is for educational purposes as part of the NLP course assignment.

## Acknowledgments

- Based on the Transformer architecture from "Attention is All You Need"
- Dataset: OPUS-100 English-Nepali corpus
- Tokenizers: spaCy and Hugging Face tokenizers