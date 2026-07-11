import pandas as pd 
from pathlib import Path
from prefect import task, flow, get_run_logger
import matplotlib.pyplot as plt 
import seaborn as sns

#Task 1:
data_dir = Path("assignments_01/resources/happiness_project")
output_dir = Path("assignments_01/outputs")
output_file = output_dir / "merged_happiness.csv"

@task(retries=3, retry_delay_seconds=2)
def load_and_merge_happiness_data():
    logger = get_run_logger()
    logger.info("Task 1: Loading all happiness CSV files...")

    all_dfs = []

    #loop all years 
    for year in range(2015, 2024 + 1):
        file_path = data_dir / f"world_happiness_{year}.csv"
        logger.info(f"Loading file: {file_path}")

        df = pd.read_csv(
            file_path,
            sep=";",
            decimal= ",",
            encoding="utf-8",
        )

        #fix different column names 
        df = df.rename(columns= {
            "Ladder score": "Happiness score"
        })

        df["Year"] = year
        all_dfs.append(df)
    
    #merge all years
    merged = pd.concat(all_dfs, ignore_index=True)

    #Make sure directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    #save datasets as merged 
    merged.to_csv(output_file, index=False)
    logger.info(f"Merged dataset saved to: {output_file}")
    return merged

#Task 2: 
@task 
def compute_descriptive_stats(df):
    logger = get_run_logger()
    logger.info("Starting Task 2: Descriptive Statistics")

    #mean, median and standard deviation overall
    mean_score = df["Happiness score"].mean()
    median_score = df["Happiness score"].median()
    std_score = df["Happiness score"].std()

    logger.info(f"Overall Mean Happiness score: {mean_score:.3f}")
    logger.info(f"Overall Median Happiness score: {median_score:.3f}")
    logger.info(f"Overall Standard Deviation Happiness score: {mean_score:.3f}")

    #mean by year 
    logger.info("Mean Happiness socre by Year:")
    mean_year = df.groupby("Year")["Happiness score"].mean()
    for year, value in mean_year.items():
        logger.info(f" {year}: {value:.3f}")
    
    #mean by region 
    logger.info("Mean Happiness score by Region:")
    mean_region = df.groupby("Regional indicator")["Happiness score"].mean()
    for region, vlaue in mean_region.items():
        logger.info(f" {region}: {value:.3f}")
    
    return {
        "overall": {
            "mean": mean_score,
            "median": median_score,
            "std": std_score,
        },
        "mean_year": mean_year, 
        "mean_region": mean_region,
    }

#Task 3: 
@task 
def create_visualizations(df):
    logger = get_run_logger()
    logger.info("Creating Visualizations")

    output_dir.mkdir(parents=True, exist_ok=True)

    #Histogram
    plt.figure(figsize=(8, 6))
    sns.histplot(df["Happiness score"], bins=20, kde=True, color="skyblue")
    plt.title("All Years Happiness Score Distribution")
    plt.xlabel("Happiness Score")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(output_dir / "happiness_histogram.png")
    plt.close()
    logger.info("Saved Histogram as happiness_histogram.png")

    #Boxplot by year
    plt.figure(figsize=(10, 6))
    sns.boxplot(x="Year", y="Happiness score", data=df)
    plt.title("By Year Happiness Score Distribution")
    plt.xlabel("Year")
    plt.ylabel("Happiness Score")
    plt.tight_layout()
    plt.savefig(output_dir / "happiness_by_year.png")
    plt.close()
    logger.info("Saved Boxplot as happiness_by_year.png")

    #Scatter plot GDP vs Happiness score
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x="GDP per capita", y="Happiness score", data=df)
    plt.title("GDP per Capita vs Happiness Score")
    plt.tight_layout()
    plt.savefig(output_dir / "gdp_vs_happiness.png")
    logger.info("Saved Scatter plot as gdp_vs_happiness.png")

    #Correlation Heatmap
    plt.figure(figsize=(10, 8))
    corr = df.select_dtypes("number").corr(method="pearson")
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(output_dir / "correlation_heatmap.png")
    plt.close()
    logger.info("Saved Heatmap as correlation_heatmap.png")

    logger.info("All visulaizations created successfully.")

#Task 4 
from scipy.stats import ttest_ind

@task 
def run_statistical_tests(df):
    logger = get_run_logger()
    logger.info("Statistical Tests")

    # Test 1: 2019 vs 2020 (Pandemic impact)
    df_2019 = df[df["Year"] == 2019]["Happiness score"].dropna()
    df_2020 = df[df["Year"] == 2020]["Happiness score"].dropna()

    t_stat, p_val = ttest_ind(df_2019, df_2020, equal_var=False)
    mean_2019 = df_2019.mean()
    mean_2020 = df_2020.mean()

    logger.info(f"T-test (2019 vs 2020): t = {t_stat:.3f}, p = {p_val:.4f}")
    logger.info(f"Mean 2019 = {mean_2019:.3f}, Mean 2020 = {mean_2020:.3f}")

    if p_val < 0.05:
        if mean_2020 < mean_2019:
            interpretation = (
                "Global happiness scores declined significantly in 2020,"
                "this suggests the pandemic negativly affected well-being."
            )
        else: 
            interpretation = (
                "Global happiness scores increased significantly in 2020,"
                "this suggests there was resilience during the pandemic."
            )
    else:
        interpretation = (
            "No statistically significant difference between 2019 and 2020 happiness scores,"
            "therefore the pandemic did not alter global happiness."
        )
    
    logger.info(f"Interpretation: {interpretation}")

    # Test 2: 
    region_a = "Western Europe"
    region_b = "East Asia"

    df_a = df[df["Regional indicator"] == region_a]["Happiness score"].dropna()
    df_b = df[df["Regional indicator"] == region_b]["Happiness score"].dropna()

    t_stat2, p_val2 = ttest_ind(df_a, df_b, equal_var=False)
    mean_a = df_a.mean()
    mean_b = df_b.mean()

    logger.info(f"T-test ({region_a} vs {region_b}: t - {t_stat2:.3f}, p = {p_val2:.4f})")
    logger.info(f"Mean {region_a} = {mean_a:.3f}, Mean {region_b} = {mean_b:.3f}")

    if p_val2 < 0.05:
        interpretation2 = (
            f"Happiness scores differ significantly between {region_a} and {region_b}. "
            f"{region_a} seems to report higher happiness, which reflects stronger social and economic indicators."
        )
    else:
        interpretation2 = (
            f"No significant difference between {region_a} and {region_b} happiness scores."
        )
        
    logger.info(f"Interpretation: {interpretation2}")

    return {
        "pandemic_test": {
            "t_stat": t_stat,
            "p_val": p_val,
            "mean_2019": mean_2019,
            "mean_2020": mean_2020,
            "interpretation": interpretation,
        },
        "regional_test": {
            "t_stat": t_stat2,
            "p_val": p_val2,
            "mean_a": mean_a,
            "mean_b": mean_b,
            "interpretation": interpretation2,
        }
    }
    

#Task 5:
from scipy.stats import pearsonr

@task
def run_correlation_analysis(df):
    logger = get_run_logger()
    logger.info("Correlation Analysis with Bonferroni Correction")

    #Numeric columns 
    numeric_columns = df.select_dtypes("number").columns.tolist()

    #remove target variable 
    numeric_columns = [col for col in numeric_columns if col != "Happiness score"]

    results = []
    logger.info(f"Computing Pearson correlations for {len(numeric_columns)} variables...")

    for col in numeric_columns: 
        clean = df[[col, "Happiness score"]].dropna()

        if len(clean) < 2:
            logger.warning(f"Not enough data to calculate sorrelation for {col}")
            continue
        
        try:
            r, p = pearsonr(clean[col], clean["Happiness score"])
            results.append((col, r, p))
            logger.info(f"{col}: r = {r:.3f}, p = {p:.4f}")
        except Exception as e: 
            logger.error(f"Correlation failed for {col}: {e}")
            continue
    #Helps in case a division by zero happens
    num_tests = len(results)
    if num_tests == 0:
        logger.error("No valid correlation tests were perfeormed. Cannot apply the Bonferroni correction.")
        return {
            "results": [], 
            "adjusted_alpha": None,
            "significant_original": [],
            "significant_corrected": [],
        }
    
    #Bonferroni correction 
    num_tests = len(results)
    adjusted_alpha = 0.05 / num_tests
    logger.info(f"Bonferroni correction applied: adjusted alpha = {adjusted_alpha:.5f}")
    
    #significant correlations
    significant_original = [col for col, r, p in results if p < 0.05]
    significant_corrected = [col for col, r, p in results if p < adjusted_alpha]

    logger.info(f"Significant at alpha = 0.05: {significant_original}")
    logger.info(f"Significant after Bonferroni correction: {significant_corrected}")

    if len(significant_corrected) < len(significant_original):
        logger.info(
            "Some correlations that appeared significant did not hold up under the Bonferroni condition."
        )
    return {
        "results": results, 
        "adjusted_alpha": adjusted_alpha,
        "significant_original": significant_original,
        "significant_corrected": significant_corrected,
    }


#Task 6: 
@task
def summary_report(stats, tests, correlations):
    logger = get_run_logger()
    logger.info("Creating summary report")

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "happiness_summary_report.txt"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("WORLD HAPPINESS PIPELINE SUMMARY REPORT\n")
        f.write("=" * 45 + "\n\n")

        #-- Task 2 For Descriptive Stats--
        f.write("TASK 2: DESCRIPTIVE STATISTICS\n")
        f.write(f"Mean Happiness Score: {stats['overall']['mean']:.3f}\n")
        f.write(f"Median Happiness Score: {stats['overall']['median']:.3f}\n")
        f.write(f"Standard Deviation Happiness Score: {stats['overall']['std']:.3f}\n\n")

        #--Task 4 for Stats Test--
        f.write("TASK 4: STATISTICAL TESTS\n")
        f.write("Pandemic Impact (2019 vs 2020)\n")
        f.write(f"t = {tests['pandemic_test']['t_stat']:.3f}, p = {tests['pandemic_test']['p_val']:.4f}\n")
        f.write(f"Mean 2019 = {tests['pandemic_test']['mean_2019']:.3f}, Mean 2020 = {tests['pandemic_test']['mean_2020']:.3f}\n")
        f.write(f"Interpretation: {tests['pandemic_test']['interpretation']}\n\n")

        f.write("Regional Comparison (Western Europe vs East Asia)\n")
        f.write(f"t = {tests['regional_test']['t_stat']:.3f}, p = {tests['regional_test']['p_val']:.4f}\n")
        f.write(f"Mean Western Europe = {tests['regional_test']['mean_a']:.3f}, Mean east Asia = {tests['regional_test']['mean_b']:.3f}\n")
        f.write(f"Interpretation: {tests['regional_test']['interpretation']}\n\n")

        #Task 5  Analysis
        f.write("TASK 5: CORRELATION ANALYSIS\n")
        f.write(f"Bonferroni-adjusted alpha: {correlations['adjusted_alpha']:.5f}\n\n")

        for col, r, p in correlations["results"]:
            f.write(f"{col}: r = {r:.3f}, p = {p:.4f}\n")
        
        f.write("\nSignificant at alpha = 0.05: \n")
        f.write(", ".join(correlations['significant_original']) + "\n")

        f.write("\nSignificant after Bonferroni correction: \n")
        f.write(", ".join(correlations['significant_corrected']) + "\n")

        f.write("\nEnd of report. \n")
    
    logger.info(f"Summary report was saved to {report_path}")



@flow 
def happiness_pipeline():
    logger = get_run_logger()
    logger.info("Running World Happiness Pipeline...")

    merged_df = load_and_merge_happiness_data()
    stats = compute_descriptive_stats(merged_df)
    create_visualizations(merged_df)
    tests = run_statistical_tests(merged_df)
    correlations = run_correlation_analysis(merged_df)
    summary_report(stats, tests, correlations)

    logger.info("Pipeline completed successfully.")
    #return merged_df
    #return stats
    #return {"stats": stats, "tests": tests, "correlations": correlations}   

if __name__ == "__main__":
    happiness_pipeline()
