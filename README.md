
# **NLP Semantic Text Similarity Project (NLP-01 Team)**

## **Project Overview**
This project focuses on building an AI model to measure the semantic similarity between two sentences using the **Semantic Text Similarity (STS)** task. The similarity is rated on a scale of 0.0 to 5.0. The primary evaluation metric used is the **Pearson Correlation Coefficient (PCC)**, and efforts have been made to improve model performance through data preprocessing, augmentation, and hyperparameter optimization.

## **Team Members**
- Park Junsung
- Lee Jaebaek
- Kang Shinu
- Hong Sungkyun
- Baek Seungwoo
- Kim Jungseok

## **Project Structure**
```bash
├── notebooks
│   └── EDA.ipynb               # Exploratory Data Analysis
├── src
│   ├── data_pipeline
│   │   ├── augment_func
│   │   │   ├── AugFunction.py  # Abstract class for augmentation functions
│   │   │   ├── swap_sentences.py  # Function to swap sentence_1 and sentence_2
│   │   │   ├── undersample_label_0.py  # Undersampling for label 0
│   │   ├── augmentation.py      # Main module for augmentation execution
│   │   ├── dataloader.py        # Data loading and tokenization
│   │   └── dataset.py           # Dataset class for input-output handling
│   ├── eda
│   │   ├── exploration.py       # Streamlit-based visualizations
│   │   └── feature.py           # Adding token length feature
│   ├── model
│   │   ├── MultiTaskLoss.py     # Multi-task loss function for regression and classification
│   │   ├── loss.py              # Custom loss functions
│   │   ├── model.py             # Model training and validation (PyTorch Lightning)
│   │   └── optimizer.py         # Custom optimizer management
│   ├── utils
│   │   ├── decorators.py        # Metadata decorators
│   │   ├── config.py            # Load YAML configurations
│   │   ├── ensemble.py          # Model ensemble logic
│   └── main.py                  # Model training, validation, and logging (PyTorch Lightning)
└── config.yaml                  # Default training and dataset configuration
```

## **Development Environment**
- **Server:** AI Stages GPU (Tesla V100-SXM2) x 4
- **Languages:** Python
- **Libraries:** Transformers, PyTorch, Pandas, WandB, Hugging Face, Matplotlib
- **Operating System:** Linux

## **How to Run**
1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-repo-url.git
   ```
2. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Configure settings:**
   Modify `config.yaml` to set the appropriate file paths and hyperparameters.

4. **Train the model:**
   ```bash
   python src/main.py
   ```

## **Data Augmentation Techniques**
We employed various augmentation techniques to improve model performance:
- **Sentence Swapping:** Swapped sentence_1 and sentence_2 in the dataset.
- **Undersampling:** Reduced the number of label_0 entries to balance the dataset.
- **Text Normalization:** Simplified repetitive syllables.
- **Translation Augmentation:** Used Google Translator to translate English to Korean.

## **Model Evaluation**
- **Metrics:** Pearson Correlation Coefficient (PCC)
- **Validation Strategy:** K-fold cross-validation was used to improve generalization.
- **Ensemble:** Model outputs were ensembled using weighted averages.

## **Results**
- **Private Leaderboard:** 6th place (PCC: 0.9391)
- **Public Leaderboard:** 9th place (PCC: 0.9341)

## **Collaborative Tools**
- **Version Control:** [GitHub](https://github.com/your-repo-url)
- **Project Management:** Notion
- **Communication:** Slack, Zoom

## **Acknowledgments**
We would like to thank our team for their hard work and collaboration throughout the project.
