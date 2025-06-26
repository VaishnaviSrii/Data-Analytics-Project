import pandas as pd
df= pd.read_excel("C:/Users/ACER/Desktop/DataAnalyticsTasks/SharkTankIndiaDataset.xlsx")
print(df.head())
print(df.shape)
print(df.columns)

#checking missing values
print(df.isnull().sum())

#cleaning text columns
df['brand_name']=df['brand_name'].astype(str).str.strip().str.title()
df['idea']=df['idea'].astype(str).str.strip().str.capitalize()

#converting numeric columns
num_cols = [
    'pitcher_ask_amount', 'ask_equity', 'ask_valuation',
    'deal_amount', 'deal_equity', 'deal_valuation',
    'amount_per_shark', 'equity_per_shark'
]

for col in num_cols:
    df[col]=df[col].replace('[\$,₹,]','', regex=True).replace(',','', regex=True).astype(float)

#handling missing values
df.fillna({
    'deal_amount': 0,
    'deal_equity': 0,
    'deal_valuation': 0,
    'ask_valuation': 0,
    'ask_equity': 0,
    'pitcher_ask_amount': 0,
    'amount_per_shark': 0,
    'equity_per_shark': 0,
    'total_sharks_invested': 0
}, inplace=True)
print(df.head())


# check logic
df['amount_ask_exceeds']=df['deal_amount']>df['ask_valuation']
df['deal_without_equity']=(df['deal']==1) & (df['deal_equity']==0)

print(df[['deal_amount', 'ask_valuation', 'amount_ask_exceeds']].head())

df['price_per_equity']=df.apply(lambda x: x['deal_amount']/x['deal_equity'] if x['deal_equity']>0 else 0, axis=1)

presence_cols = [
    'ashneer_present', 'anupam_present', 'aman_present',
    'namita_present', 'vineeta_present', 'peyush_present', 'ghazal_present'
]
df = df.drop(columns=presence_cols)
print(df)

import openpyxl
output_path=(r"C:/Users/ACER/Desktop/DataAnalyticsTasks/Cleaned_sharkTankData.xlsx")
df.to_excel(output_path, index=False)
print(f"\nCleaned dataset saved at: {output_path}")


#Top 5 most funded startups
print("\nTop 5 most funded startups:")
print(df.sort_values(by='deal_amount', ascending=False)[['brand_name', 'deal_amount']].head())

#Average deal amount
avg_deal = df['deal_amount'].mean()
print(f"\nAverage deal amount: ₹{avg_deal:.2f} Lakhs")

#Deals by each shark
print("\nDeals by each shark:")
shark_cols = ['ashneer_deal', 'anupam_deal', 'aman_deal', 'namita_deal', 'vineeta_deal', 'peyush_deal', 'ghazal_deal']
for shark in shark_cols:
    print(f"{shark.replace('_deal', '').capitalize()} deals: {int(df[shark].sum())}")

#Industry-wise deal count (by idea)
print("\nTop 10 industries (by idea):")
print(df['idea'].value_counts().head(10))

#Deal conversion rate
deal_conversion = (df['deal'].sum() / len(df)) * 100
print(f"\nOverall Deal Conversion Rate: {deal_conversion:.2f}%")

