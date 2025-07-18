import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def display_stats(stat, category):
    if category in ["Age", "BMI"]:
        print(f"""{category}:
Mean: {round(stat[0], 2)}
Median: {stat[1]}
Standard Deviation: {round(stat[2], 2)}
Maximum: {stat[3]}
Minimum: {stat[4]}
Number: {stat[-1]}
""")
    elif category == "Gender":
        print(f"""{category}:
Male: {stat[0]}
Male %: {round(stat[1], 2)}
Female: {stat[2]} 
Female %: {round(stat[3], 2)}
Ratio (M:F): 1:{round(stat[4], 2)} 
Number: {stat[-1]}       
""")
    elif category == "Comorbidities":
        print(f"""{category}:
Yes: {stat[0]}
Yes %: {round(stat[1], 2)}
No: {stat[2]}
No %: {round(stat[3], 2)}
Ratio (No:Yes): 1:{round(stat[4], 2)}
Number: {stat[-1]}
""")
    elif category == "FOV":
        print(f"""{category}:
Targeted: {stat[0]}
Targeted %: {round(stat[1], 2)}
Whole Body: {stat[2]}
Whole Body %: {round(stat[3], 2)}
Ratio (WB:T): 1:{round(stat[4], 2)}
Number: {stat[-1]}""")
    else:
        print("Category not recognised")
    
    return None

def replace_NaN(df, columns):
    for column in columns:
        if column in df.columns:
            # Replaces the single backslash character '\' or any empty/whitespace-only strings with NaN
            df[column] = df[column].replace(to_replace=[r'\\', r'^\s*$'], value=np.nan, regex=True)
    return df

def age_stats(objective_data):
    mean_age = objective_data.Age.mean()
    median_age = objective_data.Age.median()
    std_age = objective_data.Age.std()
    max_age = objective_data.Age.max()
    min_age = objective_data.Age.min()
    number = objective_data.Age.count()
    return [mean_age, median_age, std_age, max_age, min_age, number]

def bmi_stats(objective_data):
    mean_bmi = objective_data["BMI"].mean()
    median_bmi = objective_data["BMI"].median()
    std_bmi = objective_data["BMI"].std()
    max_bmi = objective_data["BMI"].max()
    min_bmi = objective_data["BMI"].min()
    number = objective_data["BMI"].count()
    return [mean_bmi, median_bmi, std_bmi, max_bmi, min_bmi, number]

def gender_stats(objective_data):
    male_count = len(objective_data[objective_data["Gender"] == "M"])
    female_count = len(objective_data[objective_data["Gender"] == "F"])
    male_to_female_ratio = female_count / male_count
    number = objective_data.Gender.count()
    male_percentage = (male_count / number) * 100
    female_percentage = (female_count / number) * 100
    return [male_count, male_percentage, female_count, female_percentage, male_to_female_ratio, number]

def comorbidity_stats(objective_data):
    number = objective_data.Comorbidities.count()
    yes_count = len(objective_data[objective_data["Comorbidities"] == "Yes"])
    yes_percentage = (yes_count / number) * 100
    no_count = len(objective_data[objective_data["Comorbidities"] == "No"])
    no_percentage = (no_count / number) * 100
    no_to_yes_ratio = yes_count / no_count
    return [yes_count, yes_percentage, no_count, no_percentage, no_to_yes_ratio, number]

def scan_stats(objective_data):
    number = objective_data["FOV"].count()
    targeted_count = len(objective_data[objective_data["FOV"] == "T"])
    targeted_percentage = (targeted_count / number) * 100
    whole_count = len(objective_data[objective_data["FOV"] == "WB"])
    whole_percentage = (whole_count / number) * 100
    whole_to_targeted_ratio = targeted_count / whole_count
    return [targeted_count, targeted_percentage, whole_count, whole_percentage, whole_to_targeted_ratio, number]

def times_between(objective_data):
    # time_difference = lambda row: 0 if (row["Date of prosthesis insertion"] == row["Date of symptom onset"]) else ((row["Date of symptom onset"] - row["Date of prosthesis insertion"]) * 12) if (type(row["Date of symptom onset"]) == type(row["Date of prosthesis insertion"]) == int)
    return

def plot_scan_outcomes(df, columns):
    # Sets the number of rows and columns for the subplots
    n_cols = 2
    n_rows = (len(columns) + 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 7))
    axes = axes.flatten() # Flatten the axes array for easy iteration

    # Define a color palette for the plots
    palettes = ['viridis', 'plasma', 'magma', 'cividis']

    for i, column in enumerate(columns):
        if column in df.columns:
            # Create the countplot
            sns.countplot(
                x=column,
                data=df,
                ax=axes[i],
                palette=palettes[i % len(palettes)], # Cycle through palettes
                order=df[column].value_counts().index,
                hue=column,
                legend=False
            )
            
            # Set titles and labels
            axes[i].set_title(f'Distribution of {column}', fontsize=14)
            axes[i].set_xlabel('Outcome', fontsize=12)
            axes[i].set_ylabel('Count', fontsize=12)
            
            # Rotate and align x-axis labels for better readability
            axes[i].tick_params(axis='x', rotation=45, labelsize=10, labelright=False)
            plt.setp(axes[i].get_xticklabels(), ha="right", rotation_mode="anchor")

        else:
            print(f"Warning: Column '{column}' not found in DataFrame.")

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
        
    plt.tight_layout(pad=3.0) # Add padding between plots
    plt.show()

def main():
    data = pd.read_csv("C:/Users/satis/OneDrive/Desktop/Barts Project/Knee and hip data.csv")
    print(data.head())
    
    # Making a seperate table for objective information
    objective_data = data[["id", "Initials", "Age", "Gender", "BMI", "Date of prosthesis insertion", "Date of symptom onset", "Date of scan", 
                           "Result of bone scan", "Ortho Decision", "Comorbidities", "FOV", "Prosthesis Location"]]
    print(objective_data.head(10))

    # Making a seperate table for outcome information
    outcome_data = data[["Negative Scan Outcome", "Positive Scan Outcome"]]
    # print(outcome_data.head())
    
    # Making a seperate table for subjective information
    subjective_data = data[["Reason for intervention", "Reason for no intervention", "Additional information"]]
    # print(subjective_data.head(10))

    # Age numerical stats
    age_information = age_stats(objective_data)
    display_stats(age_information, "Age")

    # BMI numerical stats
    bmi_information = bmi_stats(objective_data)
    display_stats(bmi_information, "BMI")

    # Gender numerical stats
    gender_information = gender_stats(objective_data)
    display_stats(gender_information, "Gender")
    
    # Comorbiditiy numerical stats
    comorbidity_information = comorbidity_stats(objective_data)
    display_stats(comorbidity_information, "Comorbidities")

    # Scan FOV numerical stats
    scan_information = scan_stats(objective_data)
    display_stats(scan_information, "FOV")

    # print(type(list(objective_data["Date of prosthesis insertion"])[0]))
    
    outcome_data = replace_NaN(outcome_data, ["Negative Scan Outcome", "Positive Scan Outcome"])
    plot_scan_outcomes(outcome_data, ["Negative Scan Outcome", "Positive Scan Outcome"])


if __name__ == "__main__":
    main()