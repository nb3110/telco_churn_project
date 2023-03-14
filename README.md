# Telco customer Churn Prediction
## Machine Learning: Classification problem - Supervised learning

<img src="./img/churn.png" alt="churn" width="40%">


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

<img src="./img/missingval.png" alt="churn" width="100%">


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



<img src="./img/tenure1.png" alt="churn" width="100%">

<br>
<br>

#### `MonthlyCharges` 

- There's concentration or churned users within the 60-110 monthly charges rate, seems relevant to segment clients correctly in order to address clients with a higher price-sensitivity in a more personalized way

anderson_normality_test (There's already clear visual evidence this feature is not normal):

- Test statistic: 170.555235072914
- Critical value at 5%: 0.656
- Data does not look Gaussian (reject H0)

<img src="./img/monthlycharges1.png" alt="churn" width="100%">

<br>
<br>

#### `TotalCharges` 

- No apparent insight regarding churn, the distribution is clearly right skewed.

anderson_normality_test (There's already clear visual evidence this feature is not normal):

- Test statistic: 346.6380297042033
- Critical value at 5%: 0.656
- Data does not look Gaussian (reject H0)

<img src="./img/monthlycharges1.png" alt="churn" width="100%">

<br>
<br>

#### `ExtraChargesEstimate` 

- Given that there's a linear relationship between `MonthlyCharges` , `tenure` and `TotalCharges`, i added this feature which captures the difference between the product of MonthlyCharges and `tenure` with `TotalCharges`, in order to drop `TotalCharges` moving forward and avoid multicollinearity
- Although not gaussian according to the anderson test due to its heavy kurtosis, its the numeric variable with the closest to a normal distribution.

anderson_normality_test:

- Test statistic: 130.869284893427
- Critical value at 5%: 0.656
- Data does not look Gaussian (reject H0)

<img src="./img/ExtraChargesEstimate1.png" alt="churn" width="100%">

<br>
<br>

### `Scaling Numeric Variables`

Given that we are going to predict churn using this numerical variables, it is recommended to scale them because many machine learning algorithms rely on distance measures between data points to make predictions. If the numerical variables have different scales, it can lead to inaccuracies in the distance measures and ultimately affect the performance of the model.

  `MinMaxScaler`: This scaler scales the data to a fixed range of 0 to 1. I will use this scaler, given that the data is positive, non-Gaussian, and mainly concentrated in the minimum and/or maximum values

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


| |index|null_%|null_count|dtype|count|mean|std|min|25%|50%|75%|max|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|0|gender|0.00000|0|int32|7043|0.50000|0.50000|0.00000|0.00000|1.00000|1.00000|1.00000|
|1|PaymentMethod|0.00000|0|int32|7043|1.57000|1.07000|0.00000|1.00000|2.00000|2.00000|3.00000|
|2|PaperlessBilling|0.00000|0|int32|7043|0.59000|0.49000|0.00000|0.00000|1.00000|1.00000|1.00000|
|3|Contract|0.00000|0|int32|7043|0.69000|0.83000|0.00000|0.00000|0.00000|1.00000|2.00000|
|4|StreamingMovies|0.00000|0|int32|7043|0.99000|0.89000|0.00000|0.00000|1.00000|2.00000|2.00000|
|5|StreamingTV|0.00000|0|int32|7043|0.99000|0.89000|0.00000|0.00000|1.00000|2.00000|2.00000|
|6|TechSupport|0.00000|0|int32|7043|0.80000|0.86000|0.00000|0.00000|1.00000|2.00000|2.00000|
|7|Churn|0.00000|0|int32|7043|0.27000|0.44000|0.00000|0.00000|0.00000|1.00000|1.00000|
|8|OnlineBackup|0.00000|0|int32|7043|0.91000|0.88000|0.00000|0.00000|1.00000|2.00000|2.00000|
|9|DeviceProtection|0.00000|0|int32|7043|0.90000|0.88000|0.00000|0.00000|1.00000|2.00000|2.00000|
|10|InternetService|0.00000|0|int32|7043|0.87000|0.74000|0.00000|0.00000|1.00000|1.00000|2.00000|
|11|MultipleLines|0.00000|0|int32|7043|0.94000|0.95000|0.00000|0.00000|1.00000|2.00000|2.00000|
|12|PhoneService|0.00000|0|int32|7043|0.90000|0.30000|0.00000|1.00000|1.00000|1.00000|1.00000|
|13|Dependents|0.00000|0|int32|7043|0.30000|0.46000|0.00000|0.00000|0.00000|1.00000|1.00000|
|14|Partner|0.00000|0|int32|7043|0.48000|0.50000|0.00000|0.00000|0.00000|1.00000|1.00000|
|15|OnlineSecurity|0.00000|0|int32|7043|0.79000|0.86000|0.00000|0.00000|1.00000|2.00000|2.00000|
|16|SeniorCitizen|0.00000|0|int64|7043|0.16000|0.37000|0.00000|0.00000|0.00000|0.00000|1.00000|
|17|tenure|0.00000|0|float64|7043|0.45000|0.34000|0.00000|0.12000|0.40000|0.76000|1.00000|
|18|MonthlyCharges|0.00000|0|float64|7043|0.46000|0.30000|0.00000|0.17000|0.52000|0.71000|1.00000|
|19|TotalCharges|0.00000|0|float64|7043|0.26000|0.26000|0.00000|0.05000|0.16000|0.44000|1.00000|
|20|ExtraChargesEstimate|0.00000|0|float64|7043|0.50000|0.09000|0.00000|0.46000|0.50000|0.54000|1.00000|

<br>
<br>
<br>

## `Feature Engineering`
<br>
<br>

### `Avoiding Multicollinearity:`

TotalCharges has a high correlation with tenure (Logical)
TotalCharges has a high correlation with Monthly charges

TotalCharges should be dropped, as it doesn't add additional info and its correlated with other variables `(TotalCharges = MontlyCharges * Tenure + ExtraChargesEstimate)`.
<br>
<br>
<img src="./img/corrmatrix.png" alt="churn" width="75%">

### Correlation of each variable vs Churn:


The following features show a very low level of correlation:

    - Multiple lines
    - Phone Service
    - ExtraChargesEstimate
    - Gender
    - StreamingTV
    - StreamingMovies
    - Internet service


<img src="./img/corrvschurn.png" alt="churn" width="75%">

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


<br><br>


<img src="./img/sampling.png" alt="churn" width="75%">


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
<br><br>
<br>

Sources:

"The Value of Keeping the Right Customers" by Harvard Business Review
(https://hbr.org/2014/10/the-value-of-keeping-the-right-customers)

"Understanding the Economics of Telecom Customer Acquisition and Retention" by McKinsey & Company (https://www.mckinsey.com/industries/telecommunications/our-insights/understanding-the-economics-of-telecom-customer-acquisition-and-retention)

"Customer Acquisition vs. Retention Costs – Statistics And Trends" by Invesp (https://www.invespcro.com/blog/customer-acquisition-retention/)

"Customer Acquisition vs. Retention: Which Costs More and What Can You Do About It?" by SuperOffice (https://www.superoffice.com/blog/customer-acquisition-vs-retention/)