# Telco customer Churn Prediction
## Machine Learning: Classification problem - Supervised learning

<img src="./img/churn.png" alt="churn" width="40%" >


---


## `Problem Statement` :
<br>

    Predict the potential churn customers based on numerical and categorical features.

<br>



Customer churn is of utmost importance for businesses due to the fact that retaining existing customers is generally more lucrative than acquiring new ones. Loyal customers are known to spend more and are also more likely to refer others to the company.

Particularly in the telecommunications industry, the cost difference between acquiring a new customer and retaining an existing one can be substantial. According to some estimates, it can be up to `five times more expensive` than retaining an existing one. This is because attracting new customers often requires significant marketing and advertising expenses, as well as the cost of providing incentives to entice them to switch from a competitor. In contrast, retaining existing customers generally involves providing good customer service, resolving any issues they may have, and providing them with incentives to stay with the company, such as loyalty programs or discounts.

Furthermore, in the telecom industry, there is often a high level of competition and low switching costs, meaning that customers can easily switch to a competitor if they are dissatisfied. Therefore, retaining existing customers is crucial for telecom companies to maintain their market share and profitability.

<br>


## `Dataset`:

The datasource analyzed is a dataset from Kaggle, which shows a snapshot of a set of customers from a telco company's in a given time.

The data set includes mainly information about:

- A Churn flag indicating if the customer is active or not.
- Services that each customer has signed up for: phone, multiple lines, internet, online security, online backup, device protection, tech support, and streaming TV and movies
- Customer account information: how long they’ve been a customer, contract, payment method, paperless billing, monthly charges, and total charges
- Demographic info about customers: gender, age range, and if they have partners and dependents
  


| |Feature|Description|
|:----|:----|:----|
|0|Customer ID|Contains customer ID|
|1|gender|Whether the customer is a male or a female|
|2|SeniorCitizen|Whether the customer is a senior citizen or not (1, 0)|
|3|Partner|Whether the customer has a partner or not (Yes, No)|
|4|Dependents|Whether the customer has dependents or not (Yes, No)|
|5|tenure|Number of months the customer has stayed with the company|
|6|PhoneService|Whether the customer has a phone service or not (Yes, No)|
|7|MultipleLines|Whether the customer has multiple lines or not (Yes, No, No phone service)|
|8|InternetService|Customer’s internet service provider (DSL, Fiber optic, No)|
|9|OnlineSecurity|Whether the customer has online security or not (Yes, No, No internet service)|
|10|OnlineBackup|Whether the customer has online backup or not (Yes, No, No internet service)|
|11|DeviceProtection|Whether the customer has device protection or not (Yes, No, No internet service)|
|12|TechSupport|Whether the customer has tech support or not (Yes, No, No internet service)|
|13|streamingTV|Whether the customer has streaming TV or not (Yes, No, No internet service)|
|14|streamingMovies|Whether the customer has streaming movies or not (Yes, No, No internet service)|
|15|Contract|The contract term of the customer (Month-to-month, One year, Two year)|
|16|PaperlessBilling|Whether the customer has paperless billing or not (Yes, No)|
|17|PaymentMethod|The customer’s payment method (Electronic check, Mailed check, Bank transfer, Credit card)|
|18|MonthlyCharges|The amount charged to the customer monthly|
|19|TotalCharges|The total amount charged to the customer|
|20|Churn|Whether the customer churned or not (Yes or No)|

<br>
<br>

## `Data Wrangling`

<br>

- 7043 observations.
- There was a small amount of null values in `TotalCharges` that were approximated using following relationship: `TotalCharges` ~ `MonthlyCharges` * `tenure`
- Other than `tenure`, `MonthlyCharges` and `TotalCharges`, all variables are categorical.
- No duplicate rows were found.

<br>

<img src="./img/missingval.png" alt="churn" width="100%" style="display: block; margin: 0 auto">



<br>
<br>


### Numerical Variables
<br>

#### `tenure` 

- distribution concentrates values on the extremes (min/max)
- There's a clear concentration on churned users on the min tenure values vs non churned users who have spikes both at min and max values
    - this suggests tenure could be a relevant feature to predict Churn

anderson_normality_test (There's already clear visual evidence this feature is not normal):

- Test statistic: 203.2354707966997
- Critical value at 5%: 0.656
- Data does not look Gaussian (reject H0)



<img src="./img/tenure1.png" alt="churn" width="100%" style="display: block; margin: 0 auto">>

<br>
<br>

#### `MonthlyCharges` 

- There's concentration or churned users within the 60-110 monthly charges rate, seems relevant to segment clients correctly in order to address clients with a higher price-sensitivity in a more personalized way

anderson_normality_test (There's already clear visual evidence this feature is not normal):

- Test statistic: 170.555235072914
- Critical value at 5%: 0.656
- Data does not look Gaussian (reject H0)

<img src="./img/monthlycharges1.png" alt="churn" width="100%" style="display: block; margin: 0 auto">>

<br>
<br>

#### `TotalCharges` 

- No apparent insight regarding churn, the distribution is clearly right skewed.

anderson_normality_test (There's already clear visual evidence this feature is not normal):

- Test statistic: 346.6380297042033
- Critical value at 5%: 0.656
- Data does not look Gaussian (reject H0)

<img src="./img/monthlycharges1.png" alt="churn" width="100%" style="display: block; margin: 0 auto">>

<br>
<br>

#### `ExtraChargesEstimate` 

- Given that there's a linear relationship between `MonthlyCharges` , `tenure` and `TotalCharges`, i added this feature which captures the difference between the product of MonthlyCharges and `tenure` with `TotalCharges`, in order to drop `TotalCharges` moving forward and avoid multicollinearity
- Although not gaussian according to the anderson test due to its heavy kurtosis, its the numeric variable with the closest to a normal distribution.

anderson_normality_test:

- Test statistic: 130.869284893427
- Critical value at 5%: 0.656
- Data does not look Gaussian (reject H0)

<img src="./img/ExtraChargesEstimate1.png" alt="churn" width="100%" style="display: block; margin: 0 auto">

<br>
<br>

### `Scaling Numeric Variables`

Given that we are going to predict churn using this numerical variables, it is recommended to scale them because many machine learning algorithms rely on distance measures between data points to make predictions. If the numerical variables have different scales, it can lead to inaccuracies in the distance measures and ultimately affect the performance of the model.

  `MinMaxScaler`: This scaler scales the data to a fixed range of 0 to 1. I will use this scaler, given that the data is positive and non-Gaussian.

<br>
<br>
<br>
<br>

### Categorical Variables

<br>

#### `Categorical Encoding`  

<br>

Categorical encoding is necessary because many machine learning algorithms cannot directly handle categorical variables, which are variables that take on a finite set of values rather than numeric values. To use categorical variables in machine learning models, they must first be transformed into a numerical representation. This process is called encoding. There are several encoding techniques available, each with its own advantages and disadvantages. 

Given that our dataset has features with max 3 categories per feature, i used One-Hot Encoding for binary features and label encoding for the rest.

`One-Hot Encoding:` This technique creates a binary vector for each category, with a 1 indicating the presence of the category and a 0 indicating the absence. One-hot encoding works well for categorical variables with a small number of categories.

`Label Encoding`: This technique assigns a unique integer value to each category. Label encoding works well for categorical variables with a large number of categories.

<br>

|Feature|Values|count_val|
|:---:|:---:|:---:|
|0|gender|[Female, Male]|2|
|1|Partner|[Yes, No]|2|
|2|Dependents|[No, Yes]|2|
|3|PhoneService|[No, Yes]|2|
|4|MultipleLines|[No phone service, No, Yes]|3|
|5|InternetService|[DSL, Fiber optic, No]|3|
|6|OnlineSecurity|[No, Yes, No internet service]|3|
|7|OnlineBackup|[Yes, No, No internet service]|3|
|8|DeviceProtection|[No, Yes, No internet service]|3|
|9|TechSupport|[No, Yes, No internet service]|3|
|10|StreamingTV|[No, Yes, No internet service]|3|
|11|StreamingMovies|[No, Yes, No internet service]|3|
|12|Contract|[Month-to-month, One year, Two year]|3|
|13|PaperlessBilling|[Yes, No]|2|
|14|PaymentMethod|[Electronic check, Mailed check, Bank transfer (automatic), Credit card (automatic)]|4|
|15|Churn|[No, Yes]|2|


<br>
<br>
<br>

## `Descriptive Statistics`

### Categorical Variables

<br>

`main takeouts`

1- Crosseling clearly contributes to churn reduction, leaving out streaming services.

2- Contract type and payment options are considerably relevant, with Month-to-Month based Contract, PaperlessBilling particularly with Electronic check seem to be churn drivers.

------------- 
#### User Demographics
<br>

<img src="./img/dashboard1.png" alt="churn" width="100%" style="display: block; margin: 0 auto">
<br>

- Gender has exacly the same proportion in each case, irrelevant feature.
- No dependants have a higher churn proportion.
- Partners have a consistent proportion for each category.
- Senior citizens have a higher churn proportion.

<br>

#### Contract/Payment
<br>
<img src="./img/dashboard2.png" alt="churn" width="100%" style="display: block; margin: 0 auto">
<br>

- Customer with a Month-to-Month based Contract churned the most in this category
- A high number of customers have switched their service provider when it comes down TechSupport
- PaperlessBilling displays a high proportion of customers churning
- Customers using Electronic check as PaymentMethod churned heavily.

<br>

#### Products suite

<br>

<img src="./img/dashboard3.png" alt="churn" width="100%" style="display: block; margin: 0 auto">

<br>

- MultipleLines has a slightly less proportion of churned users than other categories within this feature.
- InternetService: Fiber optic is clearly a pain point.
- StreamingTV and StreamingMovies show the same proportions, doesn't seem relevant for churn.
- A high proportion of customers without OnlineSecurity, OnlineBackup and DeviceProtection churned.








<br>

## `Feature Engineering`
----------------
<br>
<br>

### `Avoiding Multicollinearity:`

TotalCharges has a high correlation with tenure (Logical)
TotalCharges has a high correlation with Monthly charges

TotalCharges should be dropped, as it doesn't add additional info and its correlated with other variables `(TotalCharges = MontlyCharges * Tenure + ExtraChargesEstimate)`.
<br>
<br>
<img src="./img/corrmatrix.png" alt="churn" width="75%" style="display: block; margin: 0 auto">>

### `Correlation of each variable vs Churn`:


The following features show a very low level of correlation:

    - Multiple lines
    - Phone Service
    - ExtraChargesEstimate
    - Gender
    - StreamingTV
    - StreamingMovies
    - Internet service


<img src="./img/corrvschurn.png" alt="churn" width="75%" style="display: block; margin: 0 auto">

<br><br>

### `Testing Significance for Categorical Variables: Chi-Squared test`

The chi-squared test is a statistical test that can be used to determine if there is a relationship between two categorical variables. It is often used to test the independence between a categorical feature and a categorical target variable in a classification problem.

- `PhoneService` and `Gender` have a P-Value less than 5%, suggesting that they shouldn't be considered


| |feature|p_value|
|:---:|:---:|:---:|
|3|PhoneService|0.33878253580669281941|
|0|gender|0.48657873605618595647|

<br><br>

### `Testing Significance for Numerical Variables: ANOVA`
<br><br><br>
MISSING

### `Dropping non significant features:`
<br>

- 'MultipleLines'
- 'PhoneService'
- 'ExtraChargesEstimate'
- 'gender'
- 'StreamingTV'
- 'StreamingMovies'
- 'StreamingMovies'
- 'TotalCharges'

<br>

### `Target Variable: the imbalance problem`
<br>
This classification task has a clear problem: the number of churned customers is much smaller than the number of non-churned customers. This can lead to biased models that predict the majority class more frequently. Several techniques such as oversampling, undersampling, and cost-sensitive learning will be analyzed to mitigate this problem and improve the accuracy of the models.

4 possible options chosen: 

- Base scenario (no Over-Sampling),
- SMOTE with sampling_strategy=0.5 
- SMOTE with sampling_strategy=1
- Undersampling - Random Undersampling


<br>


<img src="./img/sampling.png" alt="churn" width="75%" style="display: block; margin: 0 auto">


<br>
<br><br>

## Modelling
------------------------

### **Measuring model performance**

*The overall accuracy of the model was be inferred from the averages of the precision, recall and f1-score metrics.*

<br>

`Precision`: the proportion of predicted positive instances that are actually positive. It measures the model's accuracy in predicting positive instances.

`Recall`: the proportion of actual positive instances that are correctly predicted as positive. It measures the model's completeness in predicting positive instances.

`F1-score`: the harmonic mean of precision and recall. It provides a balanced measure of both precision and recall.

`ROC AUC`: (Receiver Operating Characteristic Area Under Curve) is a performance metric commonly used in binary classification problems to evaluate the ability of a model to distinguish between positive and negative classes. ROC AUC is a measure of the model's ability to correctly rank true positives (TP) higher than false positives (FP) across a range of decision thresholds.


<br>

### **Machine Learning Pipeline**

Given that getting the best model possible without overfitting with such a heavy class imbalance, i iterated over several sampling, models, parameters and feature reduction options to get the best result possible. I benchmarked this with an AutoML package (H20).

This Pipeline works on the preprocessed data given the following inputs:

- **List of resamples:**
    - No resampling
    - SMOTE = 0.5
    - SMOTE = 1
    - Random undersampling

<br>

- **List of Classification Machine Learning models:**

    - LogisticRegression

        A linear model that uses a logistic function to model the probability of the positive class. Useful for its simplicity, interpretability, and speed. Can be regularized to prevent overfitting.
    
    - RandomForestClassifier

        An ensemble model that combines multiple decision trees to improve accuracy and reduce overfitting. Useful for its flexibility, interpretability, and resistance to outliers. Can handle class imbalance with weighting or sampling.
    
    - GradientBoostingClassifier

        An ensemble model that combines multiple weak models (e.g., decision trees) in a sequential manner to improve accuracy. Useful for its high accuracy, flexibility, and ability to handle different types of data. Can handle class imbalance with weighting or sampling.

    - XGBClassifier

        A gradient boosting model that uses optimized distributed gradient boosting algorithms to improve accuracy and reduce computation time. Useful for large and complex datasets, with high accuracy and good scalability. Can handle class imbalance with weighting or sampling.

    - MLPClassifier

        A neural network model that uses multiple layers of nodes to model complex non-linear relationships in the data. Useful for its flexibility, ability to handle different types of data, and high accuracy. Can handle class imbalance with weighting or resampling. Can be sensitive to the choice of hyperparameters and prone to overfitting.

<br>

- **List of Parameters** per Machine Learning model to use in Gridsearch, aligned to the classification problem at hand

- **List of Feature importance** level filtering.

<br>

### `Top 5 results`
<br>

The GradientBoostingClassifier model with Oversampling and Default parameters has the highest F1 score. As the other models, given this sample with such a low % of positive Churn cases, it struggles to get a high precision level (TruePositives / (TruePositives + FalsePositives)), although it gets a high recall value (TruePositives / (TruePositives + FalseNegatives)).

    Recall measures the ability of the model to identify all positive instances correctly. Given the problem context (Telco companies spend up to 4 times more to get a new customer than retaining one), a false negative (predicting a customer won't churn when they actually will) would probably lead to the loss of a customer and potential revenue, which would be more costly than a false positive (predicting a customer will churn when they actually won't).
<br>


| |sampling|ROC_AUC|accuracy|precision|recall|f1_score|model|params|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|1|Oversampling1|0.84050|0.74844|0.51857|0.80213|0.62991|GradientBoostingClassifier|NaN|
|2|feature_reduction|0.83920|0.73878|0.50653|0.82553|0.62783|RandomForestClassifier|0.05000|
|3|feature_reduction|0.84505|0.72743|0.49372|0.83617|0.62085|XGBClassifier|0.00000|
|4|feature_reduction|0.82998|0.76491|0.54575|0.71064|0.61738|GradientBoostingClassifier|0.01000|
|5|Undersampling|0.84594|0.72288|0.48872|0.82979|0.61514|GradientBoostingClassifier|NaN|

<br>

**AutoML (H20) Package to benchmark results**
<br>
<br>
| |sampling|ROC_AUC|accuracy|precision|recall|f1_score|model|params|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|1|Original Sample|0.8147475|0.749828|0.5231197|0.7132375|0.6014422|StackedEnsemble_|NaN|

. | .
:---------:|:------:
<img src="./img/rocauc.png" alt="churn" width="100%" style="display: block; margin: 0 auto"> |  <img src="./img/confusionmatrix.png" alt="churn" width="100%" style="display: block; margin: 0 auto">


<br>



<br><br>
<br><br>
<br><br>
<br><br>
<br><br>
<br><br>
<br><br>
<br><br>
<br><br>
<br><br>
<br><br>
<br><br>
<br>

Sources:

"The Value of Keeping the Right Customers" by Harvard Business Review
(https://hbr.org/2014/10/the-value-of-keeping-the-right-customers)

"Understanding the Economics of Telecom Customer Acquisition and Retention" by McKinsey & Company (https://www.mckinsey.com/industries/telecommunications/our-insights/understanding-the-economics-of-telecom-customer-acquisition-and-retention)

"Customer Acquisition vs. Retention Costs – Statistics And Trends" by Invesp (https://www.invespcro.com/blog/customer-acquisition-retention/)

"Customer Acquisition vs. Retention: Which Costs More and What Can You Do About It?" by SuperOffice (https://www.superoffice.com/blog/customer-acquisition-vs-retention/)