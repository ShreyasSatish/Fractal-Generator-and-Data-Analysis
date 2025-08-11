import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats

class ProcessData:

    def __init__(self, filepath=None, category=None, column=None):
        if category==None or column==None:
            try:
                self.data = pd.read_csv(filepath)
            except FileNotFoundError:
                print("File not found")
                self.data = None
        else:
            try:
                temp_data = pd.read_csv(filepath)
                self.data = temp_data[temp_data[column]==category]
            except FileNotFoundError:
                print("File not found")
                self.data = None
            except:
                print("Some other error occured")
                self.data = None

    def clean_data(self, columns=None, nan_character="\\"):
        for column in columns:
            if column in self.data:
                # Replace the desired character and any whitespace only strings with NaN
                self.data[column] = self.data[column].replace(to_replace=[r"\\", r"^\s*$"], value=np.nan, regex=True)

    def age_stats(self):
        mean_age = self.data.Age.mean()
        median_age = self.data.Age.median()
        std_age = self.data.Age.std()
        max_age = self.data.Age.max()
        min_age = self.data.Age.min()
        number = self.data.Age.count()

        print(f"""Age:
Mean: {round(mean_age), 2}
Median: {round(median_age, 2)}
Standard Deviation: {round(std_age, 2)}
Maximum: {max_age}
Minimum: {min_age}
Total: {number}
""")

        return [mean_age, median_age, std_age, max_age, min_age, number]
    
    def bmi_stats(self):
        mean_bmi = self.data["BMI"].mean()
        median_bmi = self.data["BMI"].mean()
        std_bmi = self.data["BMI"].std()
        max_bmi = self.data["BMI"].max()
        min_bmi = self.data["BMI"].min
        number = self.data["BMI"].count()

        print(f"""BMI:
Mean: {round(mean_bmi, 2)}
Median: {round(median_bmi, 2)}
Standard Deviation: {round(std_bmi, 2)}
Max: {max_bmi}
Min: {min_bmi}
Total: {number}
""")
        
        return [mean_bmi, median_bmi, std_bmi, max_bmi, min_bmi, number]
    
    def gender_stats(self):
        male_count = len(self.data[self.data["Gender"] == "M"])
        female_count = len(self.data[self.data["Gender"] == "F"])
        m_to_f_ratio = female_count / male_count
        total = self.data["Gender"].count()
        male_percentage = (male_count / total) * 100
        female_percentage = (female_count / total) * 100

        print(f"""Gender:
Male: {male_count}
Male %: {round(male_percentage, 2)}
Female: {female_count}
Female %: {round(female_percentage, 2)}
M:F Ratio: 1:{m_to_f_ratio}
Total: {total}
""")

        return [male_count, male_percentage, female_count, female_percentage, m_to_f_ratio, total]
    
    def comorbidity_stats(self, plot=True):
        total = self.data.Comorbidities.count()
        yes_count = len(self.data[self.data["Comorbidities"]== "Yes"])
        yes_percentage = (yes_count / total) * 100
        no_count = len(self.data[self.data["Comorbidities"] == "No"])
        no_percentage = (no_count / total) * 100
        n_to_y_ratio = yes_count / no_count

        print(f"""Comorbidities:
Yes: {yes_count}
Yes %: {round(yes_percentage, 2)}
No: {no_count}
No %: {round(no_percentage, 2)}
N:Y Ratio: 1:{n_to_y_ratio}
Total: {total}
""")
        
        if plot:
            plot_data = []
            comorbidity_type = self.data["Type of Comorbidity"]
            for entry in comorbidity_type:
                if "," in str(entry):
                    split = entry.split(",")
                    for condition in split:
                        plot_data.append(condition)
            fig, ax = plt.subplots()
            sns.countplot(x=plot_data, palette="viridis")
            ax.tick_params(axis="x", rotation=20, labelsize=10, labelright=False)
            ax.set_xlabel("Type of Comorbidity")
            ax.set_ylabel("Number of Occurences")
            ax.set_title("Occurences of different Comorbidities")
            manager = plt.get_current_fig_manager()
            manager.window.state("zoomed")
            plt.show()

        return [yes_count, yes_percentage, no_count, no_percentage, n_to_y_ratio, total]
    
    def scan_stats(self):
        number = self.data["FOV"].count()
        T_count = len(self.data[self.data["FOV"] == "T"])
        T_percentage = (T_count / number) * 100
        WB_count = len(self.data[self.data["FOV"] == "WB"])
        WB_percentage = (WB_count / number) * 100
        WB_to_T_ratio = T_count / WB_count

        print(f"""Scan FOV:
Targeted: {T_count}
Targeted %: {round(T_percentage, 2)}
Whole Body: {WB_count}
Whole Body %: {round(WB_percentage, 2)}
WB:T Ratio: 1:{WB_to_T_ratio}
Total: {number}""")

        return [T_count, T_percentage, WB_count, WB_percentage, WB_to_T_ratio, number]
    
    def times_between(self):
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
        insertion_dates = self.data["Date of prosthesis insertion"]
        symptom_dates = self.data["Date of symptom onsert"]
        scan_dates = self.data["Date of scan"]

        for i in range(len(insertion_dates)):
            if "-" in insertion_dates[i]:
                insertion_month, insertion_year = insertion_dates[i].split("-")
                insertion_month = months[insertion_month]
                insertion_year = int("20" + insertion_year)

                symptom_month, symptom_year = symptom_dates[i].split("-")
                symptom_month = months[symptom_month]
                symptom_year = int("20" + symptom_year)

                scan_month, scan_year = scan_dates[i].split("-")
                scan_month = months[scan_month]
                scan_year = int("20" + scan_year)

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
    elif category == "Aspiration Result":
        print(f"""{category}:
Positive: {stat[0]}
Positive %: {round(stat[1], 2)}
Negative: {stat[2]}
Negative %: {round(stat[3], 2)}
Undetermined: {stat[4]}
Undetermined %: {round(stat[5], 2)}
Ratio (P:N:U): 1:{round(stat[6], 2)}:{round(stat[7], 2)}
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

    manager = plt.get_current_fig_manager()
    manager.window.state("zoomed")
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
    manager = plt.get_current_fig_manager()
    manager.window.state("zoomed")
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
    manager = plt.get_current_fig_manager()
    manager.window.state("zoomed")
    plt.show()

    plt.figure(3)
    sns.boxplot(data=df, x="Gender", y="BMI")
    plt.title("Comparison of BMI spread with Gender")
    manager = plt.get_current_fig_manager()
    manager.window.state("zoomed")
    plt.show()

def infection_stats(infection_data):
    number = infection_data[infection_data["Aspiration Result"] != np.nan].count()[0]
    positive_count = len(infection_data[infection_data["Aspiration Result"] == "Positive"])
    positive_percentage = (positive_count / number) * 100
    negative_count = len(infection_data[infection_data["Aspiration Result"] == "Negative"])
    negative_percentage = (negative_count / number) * 100
    undertermined_count = len(infection_data[infection_data["Aspiration Result"] == "Undetermined"])
    undertermined_percentage = (undertermined_count / number) * 100
    positive_to_negative_ratio = negative_count / positive_count
    positive_to_undetermined_ratio = undertermined_count / positive_count
    fig, ax = plt.subplots()
    sns.countplot(x="Aspiration Result", data=infection_data)
    plt.xlabel("Result")
    plt.ylabel("Number of Patients")
    plt.title("Outcome of Aspiration/Biopsy")
    manager = plt.get_current_fig_manager()
    manager.window.state("zoomed")
    plt.show()
    
    return [positive_count, positive_percentage, negative_count, negative_percentage, undertermined_count, 
            undertermined_percentage, positive_to_negative_ratio, positive_to_undetermined_ratio, number]


def main():
    # data = pd.read_csv("C:/Users/satis/OneDrive/Desktop/Barts Project/Knee and hip data.csv")
    # print(data.head())
    
    # # Making a seperate table for objective information
    # objective_data = data[["id", "Initials", "Age", "Gender", "BMI", "Date of prosthesis insertion", "Date of symptom onset", "Date of scan", 
    #                        "Result of bone scan", "Ortho Decision", "Comorbidities", "FOV", "Prosthesis Location"]]
    # objective_data = replace_NaN(objective_data, ["id", "Initials", "Age", "Gender", "BMI", "Date of prosthesis insertion", "Date of symptom onset", "Date of scan", 
    #                        "Result of bone scan", "Ortho Decision", "Comorbidities", "FOV", "Prosthesis Location"])
    # print(objective_data.head(10))

    # # Making a seperate table for outcome information
    # outcome_data = data[["Negative Scan Result", "Positive Scan Result"]]
    # outcome_data = replace_NaN(outcome_data, ["Negative Scan Result", "Positive Scan Result"])
    # # print(outcome_data.head())
    
    # # Making a seperate table for subjective information
    # subjective_data = data[["Reason for intervention", "Reason for no intervention", "Additional information"]]
    # subjective_data = replace_NaN(subjective_data, ["Reason for intervention", "Reason for no intervention", "Additional information"])
    # # print(subjective_data.head(10))

    # infection_data = data[["Aspiration Result"]]
    # infection_data = replace_NaN(infection_data, ["Aspiration Result"])
    # # print(infection_data.head(10))
    # infection_information = infection_stats(infection_data)
    # display_stats(infection_information, "Aspiration Result")

    # # Age numerical stats
    # age_information = age_stats(objective_data)
    # display_stats(age_information, "Age")

    # # BMI numerical stats
    # bmi_information = bmi_stats(objective_data)
    # display_stats(bmi_information, "BMI")

    # # Gender numerical stats
    # gender_information = gender_stats(objective_data)
    # display_stats(gender_information, "Gender")
    
    # # Comorbiditiy numerical stats
    # comorbidity_information = comorbidity_stats(objective_data)
    # display_stats(comorbidity_information, "Comorbidities")

    # # Scan FOV numerical stats
    # scan_information = scan_stats(objective_data)
    # display_stats(scan_information, "FOV")

    # # Prosthetic Location numerical stats
    # prosthetic_information = prosthetic_stats(objective_data)
    # display_stats(prosthetic_information, "Prosthetic Location")
    
    # # Plotting scan outcomes and some useful demographic comparisons
    # plot_scan_outcomes(outcome_data, ["Negative Scan Result", "Positive Scan Result"])
    # plot_demographics(objective_data)

    # insertion_to_symptom_stats, insertion_to_scan_stats, symptom_to_scan_stats = times_between(objective_data)

    knees = ProcessData(filepath="C:/Users/satis/OneDrive/Desktop/Barts Project/Knee and hip data.csv", column="Prosthesis Location", category="K")
    knees.clean_data(columns=["BMI", "Negative Scan Result", "Positive Scan Result", "Aspiration Result", "Type of Comorbidity", "Surgery"])
    knees.age_stats()
    knees.bmi_stats()
    knees.gender_stats()
    knees.comorbidity_stats()

if __name__ == "__main__":
    main()