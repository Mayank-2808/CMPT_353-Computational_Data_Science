import sys
import pandas as pd
import numpy as np
from scipy import stats

OUTPUT_TEMPLATE = (
    "Initial T-test p-value: {initial_ttest_p:.3g}\n"
    "Original data normality p-values: {initial_weekday_normality_p:.3g} {initial_weekend_normality_p:.3g}\n"
    "Original data equal-variance p-value: {initial_levene_p:.3g}\n"
    "Transformed data normality p-values: {transformed_weekday_normality_p:.3g} {transformed_weekend_normality_p:.3g}\n"
    "Transformed data equal-variance p-value: {transformed_levene_p:.3g}\n"
    "Weekly data normality p-values: {weekly_weekday_normality_p:.3g} {weekly_weekend_normality_p:.3g}\n"
    "Weekly data equal-variance p-value: {weekly_levene_p:.3g}\n"
    "Weekly T-test p-value: {weekly_ttest_p:.3g}\n"
    "Mann-Whitney U-test p-value: {utest_p:.3g}"
)

def main():
    reddit_counts = pd.read_json(sys.argv[1], lines=True)

    # Filtering data for 2012 and 2013
    filtered_data = reddit_counts[(reddit_counts['date'] >= '2012-01-01') & (reddit_counts['date'] < '2014-01-01') & (reddit_counts['subreddit'] == 'canada')].copy()

    # Separating weekdays and weekends
    filtered_data['weekday'] = pd.to_datetime(filtered_data['date']).dt.weekday
    weekday_data = filtered_data[filtered_data['weekday'] < 5].copy()
    weekend_data = filtered_data[filtered_data['weekday'] >= 5].copy()

    # Initial T-test
    initial_ttest_p = stats.ttest_ind(weekday_data['comment_count'], weekend_data['comment_count']).pvalue

    # Normality tests and Levene's test
    initial_weekday_normality_p = stats.normaltest(weekday_data['comment_count']).pvalue
    initial_weekend_normality_p = stats.normaltest(weekend_data['comment_count']).pvalue
    initial_levene_p = stats.levene(weekday_data['comment_count'], weekend_data['comment_count']).pvalue

    # Transform data (using counts**2 transformation)
    transformed_weekday_data = weekday_data['comment_count']**2
    transformed_weekend_data = weekend_data['comment_count']**2

    # Normality tests and Levene's test on transformed data
    transformed_weekday_normality_p = stats.normaltest(transformed_weekday_data).pvalue
    transformed_weekend_normality_p = stats.normaltest(transformed_weekend_data).pvalue
    transformed_levene_p = stats.levene(transformed_weekday_data, transformed_weekend_data).pvalue

    # Weekly data aggregation
    filtered_data['year'] = pd.to_datetime(filtered_data['date']).dt.year
    filtered_data['week'] = pd.to_datetime(filtered_data['date']).dt.isocalendar().week
    weekly_data = filtered_data.groupby(['year', 'week', 'weekday']).agg({'comment_count': 'mean'}).reset_index()

    # Normality tests and Levene's test on weekly data
    weekly_weekday_normality_p = stats.normaltest(weekly_data[weekly_data['weekday'] < 5]['comment_count']).pvalue
    weekly_weekend_normality_p = stats.normaltest(weekly_data[weekly_data['weekday'] >= 5]['comment_count']).pvalue
    weekly_levene_p = stats.levene(weekly_data[weekly_data['weekday'] < 5]['comment_count'], weekly_data[weekly_data['weekday'] >= 5]['comment_count']).pvalue

    # Weekly T-test
    weekly_ttest_p = stats.ttest_ind(weekly_data[weekly_data['weekday'] < 5]['comment_count'], weekly_data[weekly_data['weekday'] >= 5]['comment_count']).pvalue

    # Mann-Whitney U-test on original data
    utest_p = stats.mannwhitneyu(weekday_data['comment_count'], weekend_data['comment_count'], alternative='two-sided').pvalue

    print(OUTPUT_TEMPLATE.format(
        initial_ttest_p=initial_ttest_p,
        initial_weekday_normality_p=initial_weekday_normality_p,
        initial_weekend_normality_p=initial_weekend_normality_p,
        initial_levene_p=initial_levene_p,
        transformed_weekday_normality_p=transformed_weekday_normality_p,
        transformed_weekend_normality_p=transformed_weekend_normality_p,
        transformed_levene_p=transformed_levene_p,
        weekly_weekday_normality_p=weekly_weekday_normality_p,
        weekly_weekend_normality_p=weekly_weekend_normality_p,
        weekly_levene_p=weekly_levene_p,
        weekly_ttest_p=weekly_ttest_p,
        utest_p=utest_p,
    ))

if __name__ == '__main__':
    main()

# Had to correct my error while filtering data using ChatGPT. It suggested to use the datetime conversion.
