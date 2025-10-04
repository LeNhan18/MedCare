# Medical Chatbot Project

This project is a deep learning-based medical chatbot that suggests medications based on user-reported symptoms. The chatbot utilizes a combination of natural language processing and machine learning techniques to classify symptoms and recommend appropriate medications.

## Project Structure

```
medical-chatbot
├── src
│   ├── main.py                     # Entry point of the application
│   ├── models                       # Contains model definitions
│   │   ├── symptom_classifier.py    # Class for symptom classification
│   │   ├── drug_recommender.py      # Class for medication recommendations
│   │   └── chatbot_model.py          # Integrates classifier and recommender
│   ├── data                         # Data handling and preprocessing
│   │   ├── preprocessor.py          # Functions for data preprocessing
│   │   ├── data_loader.py           # Class for loading and batching data
│   │   └── augmentation.py          # Data augmentation techniques
│   ├── training                     # Training and evaluation logic
│   │   ├── trainer.py               # Class for managing training
│   │   └── evaluator.py             # Class for model evaluation
│   ├── api                          # API setup and routes
│   │   ├── app.py                   # Flask application setup
│   │   ├── routes.py                # API routes for chatbot interaction
│   │   └── middleware.py            # Middleware for request handling
│   ├── utils                        # Utility functions
│   │   ├── text_processing.py        # Text processing utilities
│   │   ├── medical_utils.py         # Medical data handling utilities
│   │   └── config.py                # Configuration settings
│   └── chatbot                      # Chatbot conversation management
│       ├── conversation_handler.py   # Manages conversation flow
│       └── response_generator.py     # Generates responses based on input
├── data
│   ├── raw                          # Directory for raw data
│   │   └── .gitkeep
│   ├── processed                    # Directory for processed data
│   │   └── .gitkeep
│   └── models                       # Directory for model files
│       └── .gitkeep
├── notebooks                        # Jupyter notebooks for exploration and training
│   ├── data_exploration.ipynb
│   ├── model_training.ipynb
│   └── evaluation.ipynb
├── tests                            # Unit tests for the application
│   ├── test_models.py               # Tests for model classes
│   ├── test_api.py                  # Tests for API functionality
│   └── test_utils.py                # Tests for utility functions
├── frontend                         # Frontend interface files
│   ├── index.html                   # Main HTML file
│   ├── style.css                    # Styles for the frontend
│   ├── script.js                    # JavaScript for user interactions
│   └── assets                       # Directory for assets
│       └── .gitkeep
├── requirements.txt                 # Project dependencies
├── setup.py                         # Packaging and dependency management
├── config.yaml                      # Configuration settings in YAML format
├── .gitignore                       # Files to ignore in version control
└── README.md                        # Project documentation
```

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd medical-chatbot
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up the configuration in `config.yaml` as needed.

## Usage

1. Start the application:
   ```
   python src/main.py
   ```

2. Access the chatbot through the API or the frontend interface.

## Contributing

Contributions are welcome! Please submit a pull request or open an issue for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.