def Chi_sq_test(df, dependant, independant):
    #takes in the names of a dependant and independant variable (column), runs a chi squared test and then outputs 
    #a seaborn heatmap of the percent difference between the expected and actual values
    
    from scipy.stats import chi2_contingency
    from scipy.stats import chi2
    
    #create cotingency table
    count_table = df.groupby([dependant, independant])['id'].count().unstack()
    count_table.fillna(0, inplace=True)
    count_table = count_table.astype('int')
    
    #Chi Squared test is for only counts above 5, we are keeping the same ratio, but increasing min value to 5 in each column
    if count_table.isin(range(0,5)).any().any():
        for j in range(len(count_table.columns)):
            for i in range(len(count_table.index)):
                if count_table.iloc[i,j] < 1:
                    count_table.iloc[i,j] = 5
                    count_table.iloc[:,j] = count_table.iloc[:,j]*5
                elif count_table.iloc[i,j] <5:
                    count_table.iloc[:,j] = count_table.iloc[:,j]*(5/count_table.iloc[i,j])
    
    stat, p, dof, expected = chi2_contingency(count_table)
    
    #print test information
    print('P-Value = {}'.format(p))
    print('Chi Statistic = {}'.format(stat))
    print('Degrees of Freedom = {}'.format(dof))
    
    #caluclate and print heatmap
    plt.figure(figsize=(12,6))
    sns.heatmap(((count_table - expected) / count_table *100), annot=True, vmax=100, vmin=-100, fmt='.1f', 
                annot_kws={'rotation': 90}, cmap='viridis')
    plt.title('Percent Difference of Expected vs. Actual Classes per {}'.format(str.title(independant)))
    plt.show()