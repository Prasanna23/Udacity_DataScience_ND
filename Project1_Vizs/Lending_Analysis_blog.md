# Exploring Loan Delinquency with Data-Driven Insights

A deep dive into Lending Club's data to uncover trends in loan performance, borrower behavior, and regional variations.

---

## Introduction
- As part of my Udacity Data Scientist Nanodegree program, I explored Lending Club's approved and rejected loans dataset available on Kaggle. My goal was to uncover actionable insights that benefit both lenders and borrowers. Borrowers can learn strategies to improve their loan approval chances, while lenders gain a better understanding of which loans are likely to succeed.

- The dataset was already well-prepared, so minimal preprocessing was required. However, I excluded loan records with missing data for critical variables to ensure the analysis remains accurate and unbiased. The dataset includes over 150 features for approved loans and around 10 features for rejected loans.

- To focus my analysis, I first conducted an exploratory data analysis (EDA) to identify trends and patterns. Building on these findings, I delved into geographical and borrower-specific factors to examine their impact on loan performance and delinquency.

---

## Delinquency Analysis
### Data Exploration
- Exploring the LC loan data helps us to understand the nature of the accepted/funded loans.
- I wanted to look at the loans which are in one of the default statuses (***Late (31-120 days), Late (16-30 days), Charged Off, Default***). These statuses tells us that the loan is not in good standing.
    ![Loan count by status](Delinq_Statuses.png)
- By grouping different default status into one will give us a clear picture on what percentage of loans are defaulted. This is done by creating a new column "is_delinquent" and setting it to True if it has one of the default statuses. This shows that approximately 15% of the loans are delinquent. 
    ![Delinquency Percent](Delinq_Percent.png)
- Looking closely at the delinquent loans, I wanted to see if there are any specific loan characteristic that drives the delinquency rate. 
- Loan delinquency happens across all loan amount ranges but they're more prevalent around loan amounts of $10k which is seen in the density plot. 
  ![Loan-Delinquency Density plot](Delinq_Density.png)
- The loan amount spread is from $500 to $40K, plotting a box plot on loan amount vs delinquency status we can find out that 50% of the delinquent loans are having loan amount range of $8K-20K but this doesn't give us any concrete insights as 50% of non delinquent loans also fall under the same loan amount range.
   ![Loan amount Box plot](Box_Plot_Deliq.png)
- Continuing with the exploration, I wanted to check if any loan amount ranges see uptick in delinquency rates. It does show that the lower loan amounts have less deliquency rates but there's no conclusive evidence here and we will have to check the impact of other variables such as geography, credit worthiness of borrowers and interest rates at which loans are made.
   ![Delinquency rate across loan amount ranges](Delinq_rate_by_amount.png)

### Geographical Analysis
- We have borrower domicile data such as State and three digit zipcode available in the dataset. This can help us to analyze and see if there is any geographical trend exists in this dataset. This analysis will help to identify if there are any states where Lending club has more business penetration than other states in US.
- The top 5 states where most of the loans are underwritten are CA,NY,TX,FL,IL
  | Residence State | Number of loans|
  | --- | ---|
  |CA   | 314533
  |NY   | 186389
  |TX   | 186335
  |FL   | 161991
  |IL   |  91173
- To indentify where the bulk of delinquencies happen, choropleth map using the number of delinquent loans is helpful. 
  ![Choropleth map of delinquent loans](interactive_by_state.png)
- Data and map shows that higher delinquency rates are common in sun belt states except TX, another outlier is NY with higher rates of default. Maine & Iowa states have low delinquency rates.
***Highly delinquent states***
    | Residence State|  Delinquency Percent (%)|     Loan Amount|   Interest Rate|
    | --- | --- | ---| ---|
    |  ==AL== |     ==15.74==|  14686.24|  13.57
    |    ==MS==|      ==15.48==|  14750.15|  13.45
    |   ==AR==|      ==15.45==|  14094.56|  13.35
    |     OK|      15.22|  15014.52|  13.26
    |    LA|      15.16|  14831.57|  13.22|
    |   NV |     14.64|  14397.076|  13.21
    |    NY |     14.32|  14846.16|  13.26
    |  NM   |   14.23 | 14874.06|  13.16
    |  FL   |   14.02 | 14402.25|  13.17
    |  HI   |   13.96 | 15894.46|  13.79
***Low delinquency states***
|Residence State|  Delinquency Percent (%)|     Loan Amount|   Interest Rate
| --- | ---| --- | ---|
|==ME== |      ==6.45==|  14740.76|  12.82
|==IA==  |     ==7.14==|   8148.21 | 12.63
|46  |       VT|       8.43 | 13815.43 | 12.99
| ID  |     8.59|  14412.36|  13.30
| DC  |     9.45 | 15815.39 | 12.58

- I wanted to see how the average low amount value will vary by State and see how the loan amount stacks up for high and low delinquency rate states.
- By plotting another choropleth graph we can observe that average values are higher in Alaska, Virginia and Hawaii whereas it's low in Iowa, Vermont and Montana
  ![Choropleth by loan amount](loan_distribution_by_state.png)
- Iowa is a standout here, we have low delinquency rates and low average loan amount of 8K, As a lender, you can say that it's safe to fund a loan within 8K and be confident that 90% of the time it will be repaid without default. 
- Vermont closely follows IA but the loan amount average is not low like IA, it's impresssive to have low delinquency although the loan amount average is not so low. 
- Washington DC is another standout region, which is in top list of low delinquency rates and also making it to the top of the loan amount too. 
- It's very safe to fund a loan originating out of DC than TX or NJ or HI.
***High loan amount averages***

|Residence State |  Delinquency Percent (%) |     Loan Amount |   Interest Rate
| --- | --- | --- | --- |
|AK |     13.17|  17285.39|  13.32|
|        VA|      13.22|  16091.36|  13.11
|    HI|      13.95 | 15894.46|  13.79
|      MD |     13.70 | 15865.86 |  13.24
|        NJ|      13.44|  15832.75 | 12.99
|        ==DC== |      ==9.44== | ==15815.39== | 12.58
|     TX|      12.90|  15730.45|  12.99
|      MA|      12.47|  15675.63|  12.68

***Low loan amount averages***

|Residence State|  Delinquency Percent (%)|     Loan Amount|   Interest Rate|
| --- | --- | --- | --- |
|        ==IA== |      ==7.14== |  ==8148.21==|  12.63
| ==VT== |       ==8.43== | ==13815.43== | 12.99
|       MT |     11.08 | 13997.59 | 12.96
|   AR |      15.45 |  14094.56 |  13.36
|       OR  |     9.67 | 14165.03 |  12.96

#### Interest rate analysis by state
- Average interest rate is stable across US states with low being 12.6 and high being 13.8
- Interest rates in DC, IA, MA & NH are in the lower end of the spectrum whereas it's in the higher end for HI, AL, MS & AR
  ![Interest rate by State](interest_distri_by_state.png)

- Some of the high deliquent rate states are in the list for high interest rates as expected. 
- Apart from state, we have three digit zip code information available in the dataset. This shows us that there are some zip codes which looks ultra safe and some are very risky from data. 
  ***Zip codes with high delinquency rates***

| Delinquency Percent (%) | Loan Amount |   Interest Rate | Zip code (3 digit) |
 | --- | --- | --- | --- | 
| 100.00  |12000.00|  11.44|       513
|   100.00 |  21000.00|  18.24   |    516
| 100.00 | 12000.00 | 14.49    |   524
|100.00|  15000.00|  18.25 |      568
|   100.00|   9750.00|  15.80 |      643
| 100.00|   7000.00|   6.99|       682
| 100.00 | 13500.00 | 13.98|       889
|    100.00|  15000.00|  20.99|       938

***Zip codes with low delinquency rates***
| Delinquency Percent (%)|     Loan Amount|   Interest Rate| Zip code (3 digit)
 | --- | --- | --- | --- |
|0.0  |14812.50|  16.36|       009
|0.0 | 16000.00|  13.99 |      055
| 0.0 | 16244.23|  13.61 |      092
|    0.0|  19960.00|  15.44|       093
|   0.0 | 21433.33|   9.08|       095
|0.0 | 12500.00|   6.99|       202
|      0.0|  14500.00 | 13.99 |      269
|0.0 |  18555.56 | 14.17 |       340
|    0.0|  21000.00|  12.91|       348
| 0.0 |  2100.00|   9.99|       375

- Looking closely into the data shows us that these ultra safe(137 loans) and risky areas(8 loans) don't have a lot of loans disbursed and infact they're incomparable to the dataset volume.
  
### Borrower analysis

#### Objectives

- Geographical analysis gave great insights on market penetration and the impact of geo variables in the default rate.
- Moving from there, analysis on borrower data can help us understand what data points will make sure that the loan get approved and funded, what are some data fields to look for before approving to avoid defaults from the lender standpoint.
  
#### Data
- I'm going to use approved and rejected loan data to draw some insights on the objectives discussed above.
  
#### Insights

- Violin plots on the loan amount shows how the data is spread out when it's delinquent or based on loan term. But this doesn't give us any new or enough information to conclude anything.
- Loans with higher interest rates in general have higher delinquencies
- Delinquency odds can be visualized by combining loan amount, loan grade and deliquencies using heat map.
  ![Heat map Delinq Viz](delinq_heat_map_loan_grade_amount.png)
- Delinquency rates increase with grade and loan size due to lower borrower creditworthiness and repayment capacity.
  ![Heat map highlighted](heat_map_highlighted.png)
- Debt income ratio and annual income has profound impact on loan status like the loan grade. We can see that DTI increases with loan grade depreciation and annual income goes down too. Interesting the loan grades E,F,G have similar mean annual income and DTI. This makes it clear that higher loan values for grade E,F & G and inherently risky due to their lower repayment capacity
  ![Loan grade + DTI + income](Loan_Grade_DTI.png)
- With closer eyes, it's evident that fundability index is high such that if you get the loan approved, you can be almost certain that the loan will be fully funded. This is why rejected loans data is more important to figure out on what qualities are required for loan approval.
  ![Funding success](funding_success_HM.png)
- Applications with lower risk scores are getting rejected even for smaller loan amount requests.Better risk scores improve the approval odds but only until the requested amount is less than 10K. Beyod 10K rejections creep up even with better score.
  ![rejected loans](Rejected_loans_heatmap.png)
- DTI data gives additional information on rejections.Low score profiles carry huge DTI and on the other side of the spectrum the better risk score profiles carry heavy debt too which could be the reason why larger loan amounts are getting rejected here.
  ![rejected dti](rejected_loans_DTI.png)

#### Conclusion
- Higher risk score with lower DTI and annual income to support the loan request will strike gold as lenders will have trust over the lenders repayment capabilities even during an economic turmoil. 
- Lending club got greater market penetration in California, New York and Texas. 
- Loan defaults are more prevalent in sun belt states and State of NY. Delinquency rates are very low in Maine and Iowa.
- Zip code analysis can be extended to see which Zip codes are doing well and not.
- This analysis underscores the importance of combining regional, borrower, and macroeconomic data for informed lending. By incorporating these insights, lenders can design strategies that mitigate risk while expanding access in underserved regions. Future studies could enhance this by analyzing temporal trends or external factors such as economic shocks

#### Blog post
Medium blog post of my analysis can be accessed via this link - https://medium.com/@randomwonders/exploring-loan-delinquency-with-data-driven-insights-eaa188637ca8

#### Acknowledgements:
Lending Club Kaggle dataset from the below link 
https://www.kaggle.com/datasets/wordsforthewise/lending-club

  


