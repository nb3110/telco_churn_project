import pandas as pd

#Plotting
import seaborn as sns
import matplotlib.pyplot as plt

def summary_statistics(df):
    """
    This function returns summary statistics for a Pandas DataFrame input. Categorical variables will have NaNs for distribution related statistics
    """
    sum_stats_df = (df.describe().round(2)).transpose().reset_index().drop(columns="count")
    describe_df = pd.concat([df.isnull().sum()/(df.shape[0]),df.isnull().sum(),df.dtypes,pd.DataFrame(df.notnull().count(),columns=["count"])],axis=1)
    describe_df = describe_df.set_axis(["null_%","null_count","dtype","count"],axis=1).reset_index()
    return pd.merge(describe_df,sum_stats_df, how="left", on="index").sort_values("dtype").set_index("index").round(2)


################################################################
################ Normality Tests ###############################

from scipy.stats import anderson

def anderson_normality_test(df):
    """ Normality test using 5% significance. Recommended if you have a sample size larger than 5000 """
    # Perform the Anderson-Darling test
    result = anderson(df, dist='norm')

    # Print the test statistic and critical values
    print('Test statistic:', result.statistic)
    print('Critical value at 5%:', result.critical_values[1])
    
    # Interpret the results

    if result.statistic < result.critical_values[1]:
        print('Data looks Gaussian (fail to reject H0)')
    else:
        print('Data does not look Gaussian (reject H0)')




################################################################
################ Plots #########################################


def numeric_plots(var_,data_):
    """
    Plots:
    1. histplot of the numeric variable
    2. Boxplot of the numeric variable
    3. histplot of the numeric variable corresponding to churned clients
    4. histplot of the numeric variable corresponding to non churned clients
    """
    fig, axes = plt.subplots(2, 2, figsize=(18, 10))
    fig.suptitle(f'{var_} histplot, distribution and Churn split',fontsize=20)
    fig.tight_layout(pad=5.0)
    sns.histplot(ax= axes[0,0], data =data_, x=var_)
    sns.boxplot(ax= axes[0,1],data = data_, x=var_)
    sns.histplot(ax= axes[1,0], data=data_[data_["Churn"]=="Yes"] ,x=var_ ,color= "red" )
    sns.histplot(ax= axes[1,1], data=data_[data_["Churn"]=="No"] ,x=var_,color= "blue" )

    axes[0, 0].set_title(f'{var_} histplot')
    axes[0, 1].set_title(f'{var_} boxplot')
    axes[1, 0].set_title("Churned Count")
    axes[1, 1].set_title("Not Churned Count")
    plt.show()