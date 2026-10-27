SpamShield — Intelligent SMS Spam Detection

SpamShield is a machine-learning-based web application that classifies SMS messages as Spam or Legitimate (Ham). It provides instant message analysis, batch CSV classification, and model evaluation through an interactive dashboard.

Key Features

- Smart Detection: Classify individual SMS messages and view the predicted category and spam probability.
- Batch Analysis: Upload a CSV file to classify multiple messages at once.
- Downloadable Results: Export batch predictions as a CSV file for further analysis.
- Model Insights: Evaluate model performance using accuracy, precision, recall, F1-score, and a confusion matrix.
- Interactive Dashboard: View session activity and message classification summaries.
- User-Friendly Interface: Navigate the application's features through a clean Streamlit interface.

Technology Stack

- Programming Language: Python
- Web Framework: Streamlit
- Machine Learning: Scikit-learn
- Data Processing: Pandas
- Visualization: Plotly
- Dataset: UCI SMS Spam Collection

Project Structure

SpamShield/
├── app.py
├── train_model.py
├── SMSSpamCollection
├── spam_model_improved.pkl
├── spam_model.pkl
├── requirements.txt
└── README.md

Installation and Setup

1. Clone the repository

git clone YOUR_GITHUB_REPOSITORY_URL
cd SpamShield

Replace "YOUR_GITHUB_REPOSITORY_URL" with your actual GitHub repository URL.

2. Install dependencies

pip install -r requirements.txt

3. Run the application

streamlit run app.py

The application will open in your browser.

How to Use

1. Open the Smart Detection section.
2. Enter an SMS message and run the prediction.
3. Open Batch Analysis to upload a CSV containing a message column.
4. Analyze the messages and download the results.
5. Open Model Insights to review model evaluation metrics.

Dataset

The project uses the SMS Spam Collection dataset, containing labeled SMS messages categorized as spam or ham. The dataset is used for machine-learning-based text classification.

Learning Outcomes

- Understanding text classification and supervised machine learning.
- Working with SMS datasets and data preprocessing.
- Building an interactive Python web application.
- Evaluating classification models using standard performance metrics.
- Processing CSV uploads and exporting prediction results.

Future Improvements

- Add more robust input validation and error handling.
- Compare multiple classification algorithms.
- Improve detection of new and evolving spam patterns.
- Deploy the application online.

---

Project: SpamShield — Intelligent SMS Spam Detection
Category: Machine Learning / Natural Language Processing
Status: Working prototype