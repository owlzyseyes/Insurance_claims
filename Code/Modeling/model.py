import pandas as pd

# Import logit from statsmodels
from statsmodels.formula.api import logit


# Load the data
df_modeling = pd.read_csv("./car_insurance_cleaned.csv")

# drop the columns that are not needed
df_modeling.drop(
    ["gender", "age", "married", "education", "children"], axis=1, inplace=True
)

# Empty list to store model results
models = []

# Feature columns
features = df_modeling.drop(columns=["id", "outcome"]).columns

# Loop through features
for col in features:
    # Create a model
    model = logit(f"outcome ~ {col}", data=df_modeling).fit()
    # Add each model to the models list
    models.append(model)

# Empty list to store accuracies
accuracies = []

# Loop through models
for feature in range(0, len(models)):
    # Compute the confusion matrix
    conf_matrix = models[feature].pred_table()
    # True negatives
    tn = conf_matrix[0, 0]
    # True positives
    tp = conf_matrix[1, 1]
    # False negatives
    fn = conf_matrix[1, 0]
    # False positives
    fp = conf_matrix[0, 1]
    # Compute accuracy
    acc = (tn + tp) / (tn + fn + fp + tp)
    accuracies.append(acc)

# print accuracies matched with their columns
for i in range(0, len(features)):
    print(f"{features[i]}: {accuracies[i]}")

# Find the feature with the largest accuracy
max_acc = max(accuracies)
max_acc_index = accuracies.index(max_acc)
max_acc_feature = features[max_acc_index]
print(f"Max accuracy: {max_acc} with feature: {max_acc_feature}")
