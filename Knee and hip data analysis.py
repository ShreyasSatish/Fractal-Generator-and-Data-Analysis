import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats

"""Utility Functions"""
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
Number: {stat[-1]}
""")
    elif category == "Prosthetic Location":
        print(f"""{category}:
Knee: {stat[0]}
Knee %: {round(stat[1], 2)}
Hip: {stat[2]}
Hip %: {round(stat[3], 2)}
Ratio (H:K): 1:{round(stat[4], 2)}
Number: {stat[-1]}
""")
    else:
        print("Category not recognised")
    
    return None

def replace_NaN(df, columns):
    for column in columns:
        if column in df.columns:
            # Replaces the single backslash character '\' or any empty/whitespace-only strings with NaN
            df[column] = df[column].replace(to_replace=[r'\\', r'^\s*$'], value=np.nan, regex=True)
    return df


"""Statistic/Plotting Functions"""
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

def prosthetic_stats(objective_data):
    number = objective_data["Prosthesis Location"].count()
    knee_count = len(objective_data[objective_data["Prosthesis Location"] == "K"]) + 1
    knee_percentage = (knee_count / number) * 100
    hip_count = len(objective_data[objective_data["Prosthesis Location"] == "H"]) + 1
    hip_percentage = (hip_count / number) * 100
    hip_to_knee_ratio = knee_count / hip_count
    return [knee_count, knee_percentage, hip_count, hip_percentage, hip_to_knee_ratio, number]

def times_between(objective_data):
    months = {"Jan": 1,
              "Feb": 2,
              "Mar": 3,
              "Apr": 4,
              "May": 5,
              "Jun": 6,
              "Jul": 7,
              "Aug": 8,
              "Sep": 9,
              "Oct": 10,
              "Nov": 11,
              "Dec": 12}
    insertion_to_symptom = []
    insertion_to_scan = []
    symptom_to_scan = []
    insertion_dates = objective_data["Date of prosthesis insertion"]
    symptom_dates = objective_data["Date of symptom onset"]
    scan_dates = objective_data["Date of scan"]
    
    for i in range(len(insertion_dates)):
        
        if "-" in insertion_dates[i]:
            insertion_month, insertion_year = insertion_dates[i].split("-")
            insertion_month = months[insertion_month]
            insertion_year = "20" + insertion_year
            insertion_year = int(insertion_year)

            symptom_month, symptom_year = symptom_dates[i].split("-")
            symptom_month = months[symptom_month]
            symptom_year = "20" + symptom_year
            symptom_year = int(symptom_year)

            scan_month, scan_year = scan_dates[i].split("-")
            scan_month = months[scan_month]
            scan_year = "20" + scan_year
            scan_year = int(scan_year)

            if (insertion_year > symptom_year) or (insertion_year > scan_year) or (symptom_year > scan_year):
                print(f"Error at index {i}")
            
            else:
                # Calculate the time in months between dates
                insertion_to_symptom.append((symptom_year - insertion_year) * 12 + abs((symptom_month - insertion_month)))
                insertion_to_scan.append((scan_year - insertion_year) * 12 + abs((scan_month - insertion_month)))
                symptom_to_scan.append((scan_year - symptom_year) * 12 + abs((scan_month - symptom_month)))
        
        else:
            insertion_year = int(insertion_dates[i])
            symptom_year = int(symptom_dates[i])
            
            scan_month, scan_year = scan_dates[i].split("-")
            scan_month = months[scan_month]
            scan_year = "20" + scan_year
            scan_year = int(scan_year)

            if (insertion_year > symptom_year) or (insertion_year > scan_year) or (symptom_year > scan_year):
                print(f"Error at index {i}")
            
            else:
                # Calculate the time in months between dates
                insertion_to_symptom.append((symptom_year - insertion_year) * 12)
                insertion_to_scan.append((scan_year - insertion_year) * 12 + (scan_month - months["Jan"]))
                symptom_to_scan.append((scan_year - symptom_year) * 12 + (scan_month - months["Jan"]))
    
    # Calculate summary statistics
    insertion_to_symptom_stats = [np.mean(insertion_to_symptom), np.median(insertion_to_symptom),
                                  stats.mode(insertion_to_symptom), np.std(insertion_to_symptom),
                                  np.max(insertion_to_symptom), np.min(insertion_to_symptom),
                                  len(insertion_to_symptom)]
    insertion_to_scan_stats = [np.mean(insertion_to_scan), np.median(insertion_to_scan),
                               stats.mode(insertion_to_scan), np.std(insertion_to_scan),
                               np.max(insertion_to_scan), np.min(insertion_to_scan),
                               len(insertion_to_scan)]
    symptom_to_scan_stats = [np.mean(symptom_to_scan), np.median(symptom_to_scan),
                             stats.mode(symptom_to_scan), np.std(symptom_to_scan),
                             np.max(symptom_to_scan), np.min(symptom_to_scan),
                             len(symptom_to_scan)]
    
    # Plotting the dates in a histogram, and displaying key stat(s) on it as well
    fig4, [ax1, ax2, ax3] = plt.subplots(ncols=3, figsize=(17, 8))

    ax1.hist(insertion_to_symptom, color="b", rwidth=1, bins=np.arange(0, insertion_to_symptom_stats[4]))
    ax1.set_title("Time between Prosthesis insertion to Symptom onset")
    ax1.set_xlabel("Time in Months")
    ax1.set_ylabel("Number of Patients")
    ax1.axvline(insertion_to_symptom_stats[0], color="r", linestyle="--", linewidth=0.8)
    # Magic numbers to get text in spots that look nicer
    ax1.text(insertion_to_symptom_stats[0] - 11, 0.5, "Mean", rotation=90, transform=ax1.get_xaxis_text1_transform(0)[0])
    ax1.text(insertion_to_symptom_stats[0] + 1, 0.9, round(insertion_to_symptom_stats[0], 2), rotation=90, transform=ax1.get_xaxis_text1_transform(0)[0])
    
    ax2.hist(insertion_to_scan, color="g", rwidth=1, bins=np.arange(0, insertion_to_scan_stats[4]))
    ax2.set_title("Time between Prosthesis insertion to Scan date")
    ax2.set_xlabel("Time in Months")
    ax2.set_ylabel("Number of Patients")
    ax2.axvline(insertion_to_scan_stats[0], color="r", linestyle="--", linewidth=0.8)
    # Magic numbers to get text in spots that look nicer
    ax2.text(insertion_to_scan_stats[0] - 14, 0.5, "Mean", rotation=90, transform=ax2.get_xaxis_text1_transform(0)[0])
    ax2.text(insertion_to_scan_stats[0] + 1, 0.9, round(insertion_to_scan_stats[0], 2), rotation=90, transform=ax2.get_xaxis_text1_transform(0)[0])

    ax3.hist(symptom_to_scan, color="c", rwidth=1, bins=np.arange(0, symptom_to_scan_stats[4]))
    ax3.set_title("Time between Symptom onset to Scan date")
    ax3.set_xlabel("Time in Months")
    ax3.set_ylabel("Number of Patients")
    ax3.axvline(symptom_to_scan_stats[0], color="r", linestyle="--", linewidth=0.8)
    # Magic numbers to get text in spots that look nicer
    ax3.text(symptom_to_scan_stats[0] - 9, 0.5, "Mean", rotation=90, transform=ax3.get_xaxis_text1_transform(0)[0])
    ax3.text(symptom_to_scan_stats[0] + 1, 0.9, round(symptom_to_scan_stats[0], 2), rotation=90, transform=ax3.get_xaxis_text1_transform(0)[0])

    plt.show()
    
    # Displaying the statistics in a nicer format
    print(f"""Time in months between Prosthesis Insertion and Symptom onset:
Mean: {round(insertion_to_symptom_stats[0], 2)}
Median: {round(insertion_to_symptom_stats[1], 2)}
Mode: {insertion_to_symptom_stats[2][0]}
Standard Deviation: {round(insertion_to_symptom_stats[3], 2)}
Maximum: {round(insertion_to_symptom_stats[4], 2)}
Minimum: {round(insertion_to_symptom_stats[5], 2)}
Number: {insertion_to_symptom_stats[-1]}
""")
    
    print(f"""Time in months between Prosthesis Insertion to Scan occurence:
Mean: {round(insertion_to_scan_stats[0], 2)}
Median: {round(insertion_to_scan_stats[1]), 2}
Mode: {insertion_to_scan_stats[2][0]}
Standard Deviation: {round(insertion_to_scan_stats[3], 2)}
Maximum: {round(insertion_to_scan_stats[4], 2)}
Minimum: {round(insertion_to_scan_stats[5], 2)}
Number: {insertion_to_scan_stats[-1]}
""")
    
    print(f"""Time in months between Symptom onset and Scan date:
Mean: {round(symptom_to_scan_stats[0], 2)}
Median: {round(symptom_to_scan_stats[1], 2)}
Mode: {symptom_to_scan_stats[2][0]}
Standard Deviation: {round(symptom_to_scan_stats[3], 2)}
Maximum: {round(symptom_to_scan_stats[4], 2)}
Minimum: {round(symptom_to_scan_stats[5], 2)}
Number: {symptom_to_scan_stats[-1]}
""")
    
    return insertion_to_symptom_stats, insertion_to_scan_stats, symptom_to_scan_stats

def plot_scan_outcomes(df, columns):
    n_cols = 2
    n_rows = (len(columns) + 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 7))
    axes = axes.flatten() 

    palettes = ['viridis', 'plasma', 'magma', 'cividis']

    for i, column in enumerate(columns):
        if column in df.columns:
            sns.countplot(
                x=column,
                data=df,
                ax=axes[i],
                palette=palettes[i % len(palettes)], 
                order=df[column].value_counts().index,
                hue=column,
                legend=False
            )
            
            axes[i].set_title(f'Actions taken after a {column}', fontsize=14)
            axes[i].set_xlabel('Action', fontsize=12)
            axes[i].set_ylabel('Number of Patients', fontsize=12)
            
            axes[i].tick_params(axis='x', rotation=45, labelsize=10, labelright=False)
            plt.setp(axes[i].get_xticklabels(), ha="right", rotation_mode="anchor")

        else:
            print(f"Warning: Column '{column}' not found in DataFrame.")

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.figure(1)    
    plt.tight_layout(pad=3.0) 
    plt.show()

def plot_demographics(df):
    fig2, ax2 = plt.subplots()
    ax2.scatter(x=df["Age"], y=df["BMI"])
    plt.xlabel("Age")
    plt.ylabel("BMI")
    plt.title("Variance of BMI with Age")
    mask = ~np.isnan(df["Age"]) & ~np.isnan(df["BMI"])
    x = df["Age"][mask]
    y = df["BMI"][mask]
    r, p = stats.pearsonr(x=x, y=y)
    plt.text(.01, .95, "Correlation Coefficient = {:.2f}".format(r), transform=ax2.transAxes)
    plt.show()

    plt.figure(3)
    sns.boxplot(data=df, x="Gender", y="BMI")
    plt.title("Comparison of BMI spread with Gender")
    plt.show()



def main():
    data = pd.read_csv("C:/Users/satis/OneDrive/Desktop/Barts Project/Knee and hip data.csv")
    print(data.head())
    
    # Making a seperate table for objective information
    objective_data = data[["id", "Initials", "Age", "Gender", "BMI", "Date of prosthesis insertion", "Date of symptom onset", "Date of scan", 
                           "Result of bone scan", "Ortho Decision", "Comorbidities", "FOV", "Prosthesis Location"]]
    objective_data = replace_NaN(objective_data, ["id", "Initials", "Age", "Gender", "BMI", "Date of prosthesis insertion", "Date of symptom onset", "Date of scan", 
                           "Result of bone scan", "Ortho Decision", "Comorbidities", "FOV", "Prosthesis Location"])
    print(objective_data.head(10))

    # Making a seperate table for outcome information
    outcome_data = data[["Negative Scan Result", "Positive Scan Result"]]
    outcome_data = replace_NaN(outcome_data, ["Negative Scan Result", "Positive Scan Result"])
    # print(outcome_data.head())
    
    # Making a seperate table for subjective information
    subjective_data = data[["Reason for intervention", "Reason for no intervention", "Additional information"]]
    subjective_data = replace_NaN(subjective_data, ["Reason for intervention", "Reason for no intervention", "Additional information"])
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

    # Prosthetic Location numerical stats
    prosthetic_information = prosthetic_stats(objective_data)
    display_stats(prosthetic_information, "Prosthetic Location")
    
    # Plotting scan outcomes and some useful demographic comparisons
    plot_scan_outcomes(outcome_data, ["Negative Scan Result", "Positive Scan Result"])
    plot_demographics(objective_data)

    insertion_to_symptom_stats, insertion_to_scan_stats, symptom_to_scan_stats = times_between(objective_data)

if __name__ == "__main__":
    main()