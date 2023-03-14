# Telco customer Churn Prediction
## Machine Learning: Classification problem - Supervised learning

<img src="./img/churn.png" alt="churn" width="30%">


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

This sample data module tracks a fictional telco company's customer churn based on various factors
The data set includes mainly information about:

- Customers who left within the last month: the column is called Churn
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

## Missing values and duplicated rows


<img src="./img/missingval.png" alt="churn" width="80%%">











<br>
<br>



Sources:

"The Value of Keeping the Right Customers" by Harvard Business Review
(https://hbr.org/2014/10/the-value-of-keeping-the-right-customers)

"Understanding the Economics of Telecom Customer Acquisition and Retention" by McKinsey & Company (https://www.mckinsey.com/industries/telecommunications/our-insights/understanding-the-economics-of-telecom-customer-acquisition-and-retention)

"Customer Acquisition vs. Retention Costs – Statistics And Trends" by Invesp (https://www.invespcro.com/blog/customer-acquisition-retention/)

"Customer Acquisition vs. Retention: Which Costs More and What Can You Do About It?" by SuperOffice (https://www.superoffice.com/blog/customer-acquisition-vs-retention/)