# Olist Sentiment Analysis with Natural Language Processing

## About this project
This project works with Olist's Brazilian E-Commerce Public Dataset. Our goal is to use this dataset to develop a classification model, which will identify if a customer review was positive or negative. Furthermore, we wish to gather explainability from the model, in this case, applying LIME and SHAP. We provide a Dockerfile and Poetry configuration file for ease of running and reproductibility. 

## Environment Setup
It is always recommended to have a separate python environment for different projects. This projects utilizes `Python 3.11.5`. We walk you through the environment configuration with Poetry and the highly recommended Docker image. pip and Conda were failing to build the project due to unresolved dependency issues with Numba, hence, their usage is not recommend - but feel free to try.

### Docker
We provide a Docker image which runs our training script and allows you to interact with the files. Running the `docker build` command will build the Python 3.11 image, install Poetry and run `train.py`, which generates de .pkl models.

```bash
docker build -t bravium_heitor .
```

Running this Docker run command will allow you to interact with the image. Running with these `-v` flags allows you to access the files on the container locally, so that they may be persisted on your local machine.

Inside of the image, you may run `poetry run python explainability.py` to run LIME and SHAP and get the results. Other than that, you may play around with the files freely.

```bash
docker run -it --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/explainability:/app/explainability \
  -v $(pwd)/metrics:/app/metrics \
  -v $(pwd)/model:/app/model \
  -v $(pwd)/processed_csvs:/app/processed_csvs \
  bravium_heitor
```

### Poetry
Poetry is our preferred Python package manager and we recommend its use for this project. You should have it installed locally with pipx. There are plenty of [guides](https://www.sarahglasmacher.com/how-to-set-up-poetry-for-python/) available on this topic.

With poetry installed, just run

```bash
poetry install no-root
```

and the environment will be fully operational. The order in which we recommend running the codes is:

1. Getting the dataset from kaggle -> `get_kaggle_dataset.py`.
2. Following the `data_analysis.ipynb` and `data_cleaning.ipynb`.
3. Running `train.py` and `explainability.py` codes.

However, following that order is not necessary, since we've uploaded our processed .csv files to the `processed_csvs` folder.

The only file that is dependent on having a `.pkl` model on the `/models` folder is the `explainability.py` folder. As such, if you're unable to run the `train.py` script but still want to explore the code (or just want to access or model) you can download the pickle files [here](https://drive.google.com/drive/folders/1sQta4E4-mDGpDftF9fItM-u4O9BVzguk?usp=sharing)


## Exploratory Data Analysis (EDA)
During the EDA phase, our main goal is to understand the dataset's features and their relationships with eachother. We exclude multiple files and records from the dataset, either due to them not being suited for the analysis or having missing data, and save a much smaller sample of the dataset for the cleaning stage.


## Data Cleaning / Pre-Processing
Using the .csv file resulting from our EDA, we apply essential pre-processing steps on this stage, such as removing trailing whitespaces, emojis, special characters, and stemming.


## NLP Model
The model is defined on the `train.py` phase. The goal is to automatically classify reviews as positive or negative based on their text content. On this stage, we first transformed text into numerical features with TF–IDF vectorization, and then train the model.

The model is a Logistic Regression classifier, trained using GridSearchCV to find the best hyperparameters (C, penalty, class_weight). The training set and test set are split 80-20. The model is optimized for F1-score, which balances performance across classes.


## Evaluation and Explainability
To evaluate the model, we generate a classification graph, showcasing precision, recall and F1-score per class. We also save a confusion matrix (true vs predicted labels). Both are saved as .png images under the `/metrics` directory.

For explainability, we want to know why a review was classified as positive or negative. We use two complementary tools:

### LIME (Local Interpretable Model-agnostic Explanations)
LIME works on individual predictions. For a given review, it identifies the top words that influenced the classification. In the code, the explanation is converted into a matplotlib figure and saved as lime.png.

### SHAP (SHapley Additive exPlanations)
SHAP provides a more general view. Instead of only explaining one prediction, it highlights the most influential words across many reviews. The `explainability.py` file loads the trained Logistic Regression model and the TF–IDF pipeline and samples reviews, generating LIME and SHAP visualizations in the `/explainability` folder.

