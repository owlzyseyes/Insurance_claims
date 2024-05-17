import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def clean_data(df):
    # Change column type to object for column: 'id'
    df = df.astype({"id": "object"})

    # Derive column 'driving_exp_clean' from column: 'driving_experience'
    df["driving_exp_clean"] = df["driving_experience"].str.replace("y", "")

    # Drop column: 'driving_experience'
    df = df.drop(columns=["driving_experience"])

    # Replace missing values with the mean of each column in: 'credit_score'
    df["credit_score"].fillna(df["credit_score"].mean(), inplace=True)

    # Change column type to int32 for column: 'children'
    df = df.astype({"children": "int32"})

    # Drop column: 'postal_code'
    df = df.drop(columns=["postal_code"])

    # Replace missing values with the mean of each column in: 'annual_mileage' to 1 dp
    df["annual_mileage"].fillna(df["annual_mileage"].mean(), inplace=True)
    df["annual_mileage"] = df["annual_mileage"].round(1)

    # Change column types to category for selected columns
    categorical_columns = [
        "age",
        "gender",
        "driving_exp_clean",
        "education",
        "income",
        "vehicle_ownership",
        "vehicle_year",
        "married",
        "vehicle_type",
    ]
    df[categorical_columns] = df[categorical_columns].astype("category")

    return df


def remove_outliers_credit(df):
    # Calculate Q1 (25th percentile of the data) for column: 'credit_score'
    Q1 = df["credit_score"].quantile(0.25)

    # Calculate Q3 (75th percentile of the data) for column: 'credit_score'
    Q3 = df["credit_score"].quantile(0.75)

    # Calculate IQR for column: 'credit_score'
    IQR = Q3 - Q1

    # Filter rows where column: 'credit_score' is within 1.5*IQR of Q1 and Q3
    df = df[
        (df["credit_score"] >= Q1 - 1.5 * IQR) & (df["credit_score"] <= Q3 + 1.5 * IQR)
    ]

    return df


def remove_outliers_mileage(df):
    # Calculate Q1 (25th percentile of the data) for column: 'annual_mileage'
    Q1 = df["annual_mileage"].quantile(0.25)

    # Calculate Q3 (75th percentile of the data) for column: 'annual_mileage'
    Q3 = df["annual_mileage"].quantile(0.75)

    # Calculate IQR for column: 'annual_mileage'
    IQR = Q3 - Q1

    # Filter rows where column: 'annual_mileage' is within 1.5*IQR of Q1 and Q3
    df = df[
        (df["annual_mileage"] >= Q1 - 1.5 * IQR)
        & (df["annual_mileage"] <= Q3 + 1.5 * IQR)
    ]

    return df


def main():
    # Read the data from CSV file
    df = pd.read_csv("car_insurance.csv")

    # Clean the data
    df_clean = clean_data(df.copy())

    # Remove outliers in the credit_score column
    df_clean = remove_outliers_credit(df_clean)

    # Remove outliers in the annual_mileage column
    df_clean = remove_outliers_mileage(df_clean)

    # Display the cleaned data
    print(df_clean.head())


if __name__ == "__main__":
    main()
