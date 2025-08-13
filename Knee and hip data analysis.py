import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats

class ProcessData:

    def __init__(self, filepath=None, category=None, column=None):
        if category==None or column==None:
            try:
                print("in if statement")
                self.data = pd.read_csv(filepath)
            except FileNotFoundError:
                print("File not found")
                self.data = None
        else:
            try:
                temp_data = pd.read_csv(filepath)
                self.data = temp_data[temp_data[column]==category]
                if category=="H":
                    self.identifier = "Hips"
                else:
                    self.identifier = "Knees"
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
        self.age = self.data["Age"]
        
        mean_age = self.age.mean()
        median_age = self.age.median()
        std_age = self.age.std()
        max_age = self.age.max()
        min_age = self.age.min()
        number = self.age.count()

        print(f"""Age - {self.identifier}:
Mean: {round(mean_age, 2)}
Median: {round(median_age, 2)}
Standard Deviation: {round(std_age, 2)}
Maximum: {max_age}
Minimum: {min_age}
Total: {number}
""")

        return [mean_age, median_age, std_age, max_age, min_age, number]
    
    def bmi_stats(self):
        self.bmi = self.data["BMI"]
        
        mean_bmi = self.bmi.mean()
        median_bmi = self.bmi.mean()
        std_bmi = self.bmi.std()
        max_bmi = self.bmi.max()
        min_bmi = self.bmi.min()
        number = self.bmi.count()

        print(f"""BMI - {self.identifier}:
Mean: {round(mean_bmi, 2)}
Median: {round(median_bmi, 2)}
Standard Deviation: {round(std_bmi, 2)}
Max: {max_bmi}
Min: {min_bmi}
Total: {number}
""")
        
        return [mean_bmi, median_bmi, std_bmi, max_bmi, min_bmi, number]
    
    def gender_stats(self):
        self.male = self.data[self.data["Gender"] == "M"]
        self.female = self.data[self.data["Gender"] == "F"]
        
        male_count = len(self.male)
        female_count = len(self.female)
        m_to_f_ratio = female_count / male_count
        total = self.data["Gender"].count()
        male_percentage = (male_count / total) * 100
        female_percentage = (female_count / total) * 100

        print(f"""Gender - {self.identifier}:
Male: {male_count}
Male %: {round(male_percentage, 2)}
Female: {female_count}
Female %: {round(female_percentage, 2)}
M:F Ratio: 1:{round(m_to_f_ratio, 2)}
Total: {total}
""")

        return [male_count, male_percentage, female_count, female_percentage, m_to_f_ratio, total]
    
    def comorbidity_stats(self, plot=True):
        self.comorbidities_yes = self.data[self.data["Comorbidities"] == "Yes"]
        self.comorbidities_no = self.data[self.data["Comorbidities"] == "No"]
        
        total = len(self.comorbidities_yes) + len(self.comorbidities_no)
        yes_count = len(self.comorbidities_yes)
        yes_percentage = (yes_count / total) * 100
        no_count = len(self.comorbidities_no)
        no_percentage = (no_count / total) * 100
        n_to_y_ratio = yes_count / no_count

        print(f"""Comorbidities - {self.identifier}:
Yes: {yes_count}
Yes %: {round(yes_percentage, 2)}
No: {no_count}
No %: {round(no_percentage, 2)}
N:Y Ratio: 1:{n_to_y_ratio}
Total: {total}
""")
        
        if plot:
            plot_data = []
            self.comorbidity_type = self.data["Type of Comorbidity"]
            for entry in self.comorbidity_type:
                if isinstance(entry, str) and "," in entry:
                    split_entries = entry.split(",")
                    for condition in split_entries:
                        plot_data.append(condition.strip())
                elif isinstance(entry, str):
                    plot_data.append(entry.strip())

            comorbidity_counts = pd.Series(plot_data).value_counts()
            fig, ax = plt.subplots()
            sns.barplot(x=comorbidity_counts.index, y=comorbidity_counts.values, palette="viridis")
            ax.tick_params(axis="x", rotation=20, labelsize=10, labelright=False)
            ax.set_xlabel("Type of Comorbidity")
            ax.set_ylabel("Number of Occurences")
            ax.set_title(f"Occurences of different Comorbidities - {self.identifier}")
            manager = plt.get_current_fig_manager()
            manager.window.state("zoomed")
            plt.tight_layout()
            plt.show()

        return [yes_count, yes_percentage, no_count, no_percentage, n_to_y_ratio, total]
    
    def scan_stats(self):
        self.scan_T = self.data[self.data["FOV"] == "T"]
        self.scan_WB = self.data[self.data["FOV"] == "WB"]
        
        number = len(self.scan_T) + len(self.scan_WB)
        T_count = len(self.scan_T)
        T_percentage = (T_count / number) * 100
        WB_count = len(self.scan_WB)
        WB_percentage = (WB_count / number) * 100
        WB_to_T_ratio = T_count / WB_count

        print(f"""Scan FOV - {self.identifier}:
Targeted: {T_count}
Targeted %: {round(T_percentage, 2)}
Whole Body: {WB_count}
Whole Body %: {round(WB_percentage, 2)}
WB:T Ratio: 1:{WB_to_T_ratio}
Total: {number}""")

        return [T_count, T_percentage, WB_count, WB_percentage, WB_to_T_ratio, number]
    
    def times_between(self, plot=True):
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
        symptom_dates = self.data["Date of symptom onset"]
        scan_dates = self.data["Date of scan"]

        for i in range(len(insertion_dates)):
            if "-" in insertion_dates.iloc[i]:
                insertion_month, insertion_year = insertion_dates.iloc[i].split("-")
                insertion_month = months[insertion_month]
                insertion_year = int("20" + insertion_year)

                symptom_month, symptom_year = symptom_dates.iloc[i].split("-")
                symptom_month = months[symptom_month]
                symptom_year = int("20" + symptom_year)

                scan_month, scan_year = scan_dates.iloc[i].split("-")
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
                insertion_year = int(insertion_dates.iloc[i])
                symptom_year = int(symptom_dates.iloc[i])
            
                scan_month, scan_year = scan_dates.iloc[i].split("-")
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

        if plot:
            fig4, [ax1, ax2, ax3] = plt.subplots(ncols=3, figsize=(17,8))

            ax1.hist(insertion_to_symptom, color="b", rwidth=1, bins=np.arange(0, insertion_to_symptom_stats[4]))
            ax1.set_title(f"Time between Prosthesis insertion to Symptom onset - {self.identifier}")
            ax1.set_xlabel("Time in Months")
            ax1.set_ylabel("Number of Patients")
            ax1.axvline(insertion_to_symptom_stats[0], color="r", linestyle="--", linewidth=0.8)
            # Magic numbers to get text in spots that look nicer
            ax1.text(insertion_to_symptom_stats[0] - 11, 0.5, "Mean", rotation=90, transform=ax1.get_xaxis_text1_transform(0)[0])
            ax1.text(insertion_to_symptom_stats[0] + 1, 0.9, round(insertion_to_symptom_stats[0], 2), rotation=90, transform=ax1.get_xaxis_text1_transform(0)[0])

            ax2.hist(insertion_to_scan, color="g", rwidth=1, bins=np.arange(0, insertion_to_scan_stats[4]))
            ax2.set_title(f"Time between Prosthesis insertion to Scan date - {self.identifier}")
            ax2.set_xlabel("Time in Months")
            ax2.set_ylabel("Number of Patients")
            ax2.axvline(insertion_to_scan_stats[0], color="r", linestyle="--", linewidth=0.8)
            # Magic numbers to get text in spots that look nicer
            ax2.text(insertion_to_scan_stats[0] - 14, 0.5, "Mean", rotation=90, transform=ax2.get_xaxis_text1_transform(0)[0])
            ax2.text(insertion_to_scan_stats[0] + 1, 0.9, round(insertion_to_scan_stats[0], 2), rotation=90, transform=ax2.get_xaxis_text1_transform(0)[0])

            ax3.hist(symptom_to_scan, color="c", rwidth=1, bins=np.arange(0, symptom_to_scan_stats[4]))
            ax3.set_title(f"Time between Symptom onset to Scan date: {self.identifier}")
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
        print(f"""Time in months between Prosthesis Insertion and Symptom onset - {self.identifier}:
Mean: {round(insertion_to_symptom_stats[0], 2)}
Median: {round(insertion_to_symptom_stats[1], 2)}
Mode: {insertion_to_symptom_stats[2][0]}
Standard Deviation: {round(insertion_to_symptom_stats[3], 2)}
Maximum: {round(insertion_to_symptom_stats[4], 2)}
Minimum: {round(insertion_to_symptom_stats[5], 2)}
Number: {insertion_to_symptom_stats[-1]}
""")
    
        print(f"""Time in months between Prosthesis Insertion to Scan occurence - {self.identifier}:
Mean: {round(insertion_to_scan_stats[0], 2)}
Median: {round(insertion_to_scan_stats[1], 2)}
Mode: {insertion_to_scan_stats[2][0]}
Standard Deviation: {round(insertion_to_scan_stats[3], 2)}
Maximum: {round(insertion_to_scan_stats[4], 2)}
Minimum: {round(insertion_to_scan_stats[5], 2)}
Number: {insertion_to_scan_stats[-1]}
""")
    
        print(f"""Time in months between Symptom onset and Scan date - {self.identifier}:
Mean: {round(symptom_to_scan_stats[0], 2)}
Median: {round(symptom_to_scan_stats[1], 2)}
Mode: {symptom_to_scan_stats[2][0]}
Standard Deviation: {round(symptom_to_scan_stats[3], 2)}
Maximum: {round(symptom_to_scan_stats[4], 2)}
Minimum: {round(symptom_to_scan_stats[5], 2)}
Number: {symptom_to_scan_stats[-1]}
""")
        
        return insertion_to_scan_stats, insertion_to_symptom_stats, symptom_to_scan_stats

    def infection_stats(self, plot=True):
        self.infection = self.data[self.data["Aspiration Result"] != np.nan]
        
        number = len(self.infection)
        positive_count = len(self.infection[self.infection["Aspiration Result"] == "Positive"])
        positive_percentage = (positive_count / number) * 100
        negative_count = len(self.infection[self.infection["Aspiration Result"] == "Negative"])
        negative_percentage = (negative_count / number) * 100
        undetermined_count = len(self.infection[self.infection["Aspiration Result"] == "Undetermined"])
        undetermined_percentage = (undetermined_count / number) * 100
        positive_to_negative_ratio = negative_count / positive_count
        positive_to_undetermined_ratio = undetermined_count / positive_count

        print(f"""Aspiration Results - {self.identifier}:
Positive: {positive_count}
Positive %: {round(positive_percentage, 2)}
Negative: {negative_count}
Negative %: {round(negative_percentage, 2)}
Undetermined: {undetermined_count}
Undetermined %: {round(undetermined_percentage, 2)}
P:N:U Ratio: 1:{round(positive_to_negative_ratio, 2)}:{round(positive_to_undetermined_ratio, 2)}
Total: {number}""")
        
        if plot:
            fig, ax = plt.subplots()
            sns.countplot(x="Aspiration Result", data=self.infection)
            plt.xlabel("Aspiration Result")
            plt.ylabel("Number of Patients")
            plt.title(f"Outcome of Aspiration/Biopsy - {self.identifier}")
            manager = plt.get_current_fig_manager()
            manager.window.state("zoomed")
            plt.show()

        return [positive_count, positive_percentage, negative_count, negative_percentage, undetermined_count, undetermined_percentage, positive_to_negative_ratio, positive_to_undetermined_ratio, number]

    def outcomes_plot(self, columns=["Negative Scan Result", "Positive Scan Result"]):
        self.outcome_data = self.data[["Negative Scan Result", "Positive Scan Result"]]

        n_cols = 2
        n_rows = 1
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 7))
        axes = axes.flatten()
        palettes = ["viridis", "plasma", "magma", "cividis"]

        for i, column in enumerate(columns):
            if column in self.outcome_data.columns:
                sns.countplot(
                    x=column,
                    data=self.outcome_data,
                    ax=axes[i],
                    palette=palettes[i % len(palettes)],
                    order=self.outcome_data[column].value_counts().index,
                    hue=column,
                    legend=False
                )

                axes[i].set_title(f"Actions taken after a {column} - {self.identifier}", fontsize=14)
                axes[i].set_xlabel("Action", fontsize=12)
                axes[i].set_ylabel("Number of Patients", fontsize=12)
                axes[i].tick_params(axis="x", rotation=45, labelsize=10, labelright=False)
                plt.setp(axes[i].get_xticklabels(), ha="right", rotation_mode="anchor")
            
            else:
                print(f"Warning: Column {column} not found in DataFrame")

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



def main():

    knees = ProcessData(filepath="C:/Users/satis/OneDrive/Desktop/Barts Project/Knee and hip data.csv", column="Prosthesis Location", category="K")
    knees.clean_data(columns=["BMI", "Negative Scan Result", "Positive Scan Result", "Aspiration Result", "Type of Comorbidity", "Surgery"])
    knees.age_stats()
    knees.bmi_stats()
    knees.gender_stats()
    knees.comorbidity_stats()
    knees.scan_stats()
    knees.times_between()
    knees.infection_stats()
    knees.outcomes_plot()

    hips = ProcessData(filepath="C:/Users/satis/OneDrive/Desktop/Barts Project/Knee and hip data.csv", column="Prosthesis Location", category="H")
    hips.clean_data(columns=["BMI", "Negative Scan Result", "Positive Scan Result", "Aspiration Result", "Type of Comorbidity", "Surgery"])
    hips.age_stats()
    hips.bmi_stats()
    hips.gender_stats()
    hips.comorbidity_stats()
    hips.scan_stats()
    hips.times_between()
    hips.infection_stats()
    hips.outcomes_plot()

if __name__ == "__main__":
    main()