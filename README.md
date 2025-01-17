# Fake News Detection using Flask and Machine Learning

## Overview
This project implements a web-based fake news detection system using Flask and machine learning. The application takes news text as input and classifies it as either fake or real news using natural language processing techniques and a Passive Aggressive Classifier.

## Features
- Flask web interface for news input
- Text preprocessing with NLTK
- TF-IDF vectorization
- Passive Aggressive Classifier for prediction
- Real-time prediction capabilities
- Simple and intuitive UI

## Requirements
```
flask
pandas
sklearn
numpy
seaborn
nltk
pickle
matplotlib
```

## Project Structure
```
fake-news-detection/
├── static/
├── templates/
│   ├── index.html
│   └── prediction.html
├── app.py
├── model.pkl
├── vector.pkl
└── README.md
```

## Installation
1. Clone the repository:
```bash
git clone https://github.com/yourusername/fake-news-detection.git
cd fake-news-detection
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download NLTK data:
```python
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
```

## How It Works

### Text Preprocessing
The system processes input text through several steps:
1. Special character removal using regex
2. Text conversion to lowercase
3. Tokenization using NLTK
4. Stopword removal
5. Lemmatization using WordNetLemmatizer

### Model Details
- Model Type: Passive Aggressive Classifier
- Vectorization: TF-IDF Vectorizer
- Pre-trained models stored in:
  - `model.pkl`: Trained classifier
  - `vector.pkl`: Fitted TF-IDF vectorizer

## Usage

### Running the Application
1. Start the Flask server:
```bash
python app.py
```

2. Access the application through your web browser at `http://localhost:5000`

3. Enter the news text in the input field and submit for prediction

### API Endpoints
- `/`: Home page with input form
- `/predict`: POST endpoint for news prediction

## Project Components
1. `app.py`: Main Flask application file containing:
   - Route handlers
   - Text preprocessing logic
   - Model prediction implementation

2. Templates:
   - `index.html`: Home page with input form
   - `prediction.html`: Results page showing prediction

## Development
- The application runs in debug mode by default
- Model and vectorizer are loaded once at startup
- Preprocessing pipeline is integrated into the prediction route

## Error Handling
The application includes basic error handling:
- Invalid requests return an error message
- POST-only predictions
- Default error message for failed predictions

## Contributing
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Contact
Your Name - parmaryogin04@gmail.com
Project Link: https://github.com/123yogin/Fake-News-Detection-using-MachineLearning
