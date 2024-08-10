import pandas as pd
from scipy.stats import f_oneway, ttest_ind
from statsmodels.stats.multicomp import pairwise_tukeyhsd

def main():

    data = pd.read_csv('data.csv')

    anova_pvalue = f_oneway(*[data[data['Implementation'] == algo]['Sorting Time(s)'] for algo in data['Implementation'].unique()]).pvalue
    print(f"ANOVA p-value: {anova_pvalue}")

    implementations = data['Implementation'].unique()
    ttest_pvalues = {}

    for i, algo1 in enumerate(implementations):
        for algo2 in implementations[i+1:]:
            ttest_pvalue = ttest_ind(data[data['Implementation'] == algo1]['Sorting Time(s)'],
                                      data[data['Implementation'] == algo2]['Sorting Time(s)']).pvalue
            ttest_pvalues[f"{algo1} vs {algo2}"] = ttest_pvalue
            print(f"{algo1} vs {algo2}: {ttest_pvalue:.4f}")

    # Tukey HSD
    tukey_results = pairwise_tukeyhsd(data['Sorting Time(s)'], data['Implementation'])
    print(tukey_results)

    # Distinguishable groups based on Tukey HSD
    distinguishable_pairs = tukey_results.reject
    print(f"\nDistinguishable algorithms:\n{set(distinguishable_pairs)}")

if __name__ == "__main__":
    main()
