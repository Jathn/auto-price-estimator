# auto-price-estimator

This project was made as part of codecademies ML/AI-engineer career-path as the final project.

## Project idea:

The main idea was to create a ML-pipeline which could take in some info about a car and predict it's price.
A ML-pipeline for predicting the price of a car could even be a very useful tool for car buyers and sellers alike, and although this model has it's fair share of
imperfectness it does a good job on cars of the price range it was trained on.

## Implementation:

So the project is mainly divided into 4 parts after finding an appropriate dataset:

1. Loading the data, cleaning it and selecting the relevant features.
2. Taking that data and finding a good model and the correct parameters for it.
3. Proceeding with the chosen model to create a pipeline which implements the ML-workflow implemented in steps 1 & 2.
4. Creating a simple front-end app to display the model's capabilities.

The 3 first steps where all done in their own Jupyter Notebook. There is more detailed information inside them.
The app-implementation can be found in APPA.py.

## End result

A working pipeline which can predict the prices of used cars in the range $5000-30000 with a high accuracy and relatively low error. It performs poorly when cars would
be priced higher than this. That is because the cars in it's training data were all within this range.

Because of the simplicity of the gradio library I also managed to create a working front-end for the application.

<img width="566" alt="image" src="https://user-images.githubusercontent.com/124161756/232327218-0eb074d7-3ff4-4d27-8d67-f81ef3f62ab6.png">

Quite simple, yet took almost no time to implement.

## Trying out the model:

The pipeline is not uploaded here, since it is large in size (about 1.29 GB)
It is however available as a file at https://huggingface.co/Jathn/CarPriceX in the "files" section named "app_pipeline.pkl". From there you can download it and with
the "APPA.py" file in the same directory you should be able to run the app. Also check out requirements.txt.

Original dataset: https://www.kaggle.com/datasets/austinreese/craigslist-carstrucks-data
