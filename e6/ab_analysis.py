import sys
import pandas as pd
from scipy.stats import chi2_contingency, mannwhitneyu

OUTPUT_TEMPLATE = (
    '"Did more/less users use the search feature?" p-value:  {more_users_p:.3g}\n'
    '"Did users search more/less?" p-value:  {more_searches_p:.3g}\n'
    '"Did more/less instructors use the search feature?" p-value:  {more_instr_p:.3g}\n'
    '"Did instructors search more/less?" p-value:  {more_instr_searches_p:.3g}'
)


def chi_squared_test(data):
    
    contingency_table = pd.crosstab(data['uid'] % 2, data['search_count'] > 0)

    _, p_value, _, _ = chi2_contingency(contingency_table)

    return p_value


def mannwhitneyu_test(data):

    group_1 = data[data['uid'] % 2 == 0]['search_count']
    group_2 = data[data['uid'] % 2 == 1]['search_count']

    _, p_value = mannwhitneyu(group_1, group_2, alternative='two-sided')

    return p_value


def main():
    
    searchdata_file = sys.argv[1]

    data = pd.read_json(searchdata_file, orient='records', lines=True)

    instr_data = data[data['is_instructor']]

    more_users_p = chi_squared_test(data)
    
    searched_data = data[data['search_count'] > 0]
    more_searches_p = mannwhitneyu_test(searched_data)

    more_instr_p = chi_squared_test(instr_data)
    
    instr_searched_data = instr_data[instr_data['search_count'] > 0]
    more_instr_searches_p = mannwhitneyu_test(instr_searched_data)

    # Output
    print(OUTPUT_TEMPLATE.format(
        more_users_p=more_users_p,
        more_searches_p=more_searches_p,
        more_instr_p=more_instr_p,
        more_instr_searches_p=more_instr_searches_p,
    ))


if __name__ == '__main__':
    main()

