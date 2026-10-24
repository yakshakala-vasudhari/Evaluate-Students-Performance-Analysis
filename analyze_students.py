from pathlib import Path
import pandas as pd
import numpy as np


# Find the CSV file in this project folder
data_file = Path(__file__).parent / "student-mat.csv"


def summarize_column(df, column):
    """Calculate basic statistics for a DataFrame column."""

    if column not in df.columns:
        raise KeyError(f"Column '{column}' was not found.")

    values = pd.to_numeric(
        df[column],
        errors="coerce"
    ).dropna()

    if values.empty:
        raise ValueError(
            f"Column '{column}' contains no numeric values."
        )

    return {
        "count": int(values.count()),
        "mean": round(float(values.mean()), 2),
        "median": round(float(values.median()), 2),
    }


def main():
    df = pd.read_csv(data_file, sep=";")
    import matplotlib.pyplot as plt
    
    # Count students at each final grade.
    grade_counts = df["G3"].value_counts().sort_index()

    # Create a bar chart.
    grade_counts.plot(kind="bar", figsize=(10, 5))

    plt.title("Distribution of Final Grades")
    plt.xlabel("Final Grade (G3)")
    plt.ylabel("Number of Students")
    plt.tight_layout()
    plt.show()

    print("DATASET SHAPE")
    print(df.shape)

    print("\nFIRST FIVE ROWS")
    print(df.head())

    print("\nCOLUMN NAMES")
    print(df.columns.tolist())

    print("\nTOTAL MISSING VALUES")
    print(df.isnull().sum().sum())

    print("\nFINAL GRADE SUMMARY")
    grade_summary = summarize_column(df, "G3")
    print(grade_summary)

    print("\nSTUDY TIME SUMMARY")
    result = summarize_column(df, "studytime")
    print(result)

    print("\nFINAL GRADE DISTRIBUTION")

    print(
        df["G3"]
        .value_counts()
        .sort_index()
    )

    print("\nFINAL GRADE PERCENTAGES")

    grade_percentages = (
        df["G3"]
        .value_counts(normalize=True)
        .sort_index()
        .mul(100)
        .round(1)
    )

    print(grade_percentages)


    print("\nGRADE THRESHOLD ANALYSIS")

    valid_grades = df["G3"].dropna()

    total_students = len(valid_grades)

    students_at_least_10 = (
        valid_grades >= 10
    ).sum()

    proportion = (
        students_at_least_10 / total_students
    )

    percentage = proportion * 100

    print("Students with valid grades:", total_students)
    print("Students scoring at least 10:", students_at_least_10)
    print("Proportion:", round(proportion, 3))
    print("Percentage:", round(percentage, 1), "%")

    print("\nSTUDY-TIME DISTRIBUTION")

    studytime_counts = (
        df["studytime"]
        .value_counts()
        .sort_index()
    )

    print(studytime_counts)


    studytime_percentages = (
        df["studytime"]
        .value_counts(normalize=True)
        .sort_index()
        .mul(100)
        .round(1)
    )

    print("\nSTUDY-TIME PERCENTAGES")
    print(studytime_percentages)


    print(
        df.groupby("studytime")["G3"]
        .mean()
        .round(2)
    )

    most_frequent_study_time_category = (
        df["studytime"]
        .mode()
        .iloc[0]
    )
    print(
        f"\nMost frequent study time category: {most_frequent_study_time_category}"
    )

    
    student_features = df.loc[
    0,
    ["studytime", "absences", "G3"]
    ]
    print(student_features)


    mean_by_studytime = df.groupby("studytime")["G3"].mean()
    
    mean_by_studytime.plot(kind="bar", figsize=(8, 5))
    
    plt.title("Mean Final Grade by Study-Time Category")
    plt.xlabel("Study-Time Category")
    plt.ylabel("Mean Final Grade")
    plt.tight_layout()
    plt.show()


    df.boxplot(column="G3", by="studytime", figsize=(8, 5))
    
    plt.title("Final-Grade Distribution by Study-Time Category")
    plt.suptitle("")
    plt.xlabel("Study-Time Category")
    plt.ylabel("Final Grade")
    plt.tight_layout()
    plt.show()


    print("\nMean final grade by study-time category:")
    print(mean_by_studytime.round(2))




if __name__ == "__main__":
    main()


