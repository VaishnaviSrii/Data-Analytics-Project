import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.model_selection import train_test_split


df=pd.read_excel("C:/Users/ACER/Desktop/DataAnalyticsTasks/Airbnb_Data.xlsx")
print(df.head())


print(df.shape)
print(df.columns)

# Check missing values
print(df.isnull().sum())

# Convert review date to datetime
df['first_review'] = pd.to_datetime(df['first_review'], errors='coerce')

# Extract month
df['review_month'] = df['first_review'].dt.month

# Extract season
def get_season(month):
    if pd.isnull(month):
        return 'unknown'
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Fall'

df['review_season'] = df['review_month'].apply(get_season)
df['review_season'] = df['review_season'].str.lower()


#dropping irrelevant columns
df.drop(['id', 'name', 'description', 'first_review', 'last_review', 'host_since',
         'thumbnail_url'], axis=1, inplace=True)


#handling missing values of numeric columns and fil it with median
df['bathrooms'] = df['bathrooms'].fillna(df['bathrooms'].median())
df['review_scores_rating'] = df['review_scores_rating'].fillna(df['review_scores_rating'].median())
df['bedrooms'] = df['bedrooms'].fillna(df['bedrooms'].median())
df['beds'] = df['beds'].fillna(df['beds'].median())

# missing values of categorical column
df['host_response_rate'] = df['host_response_rate'].fillna(df['host_response_rate'].mode()[0])
df['zipcode'] = df['zipcode'].fillna('unknown')

# Convert host_response_rate from % to float
df['host_response_rate'] = df['host_response_rate'].astype(str).str.replace('%', '').astype(float)

#converting data types -- 't' to  & 'f' to '0'
bool_cols = ['host_has_profile_pic', 'host_identity_verified', 'instant_bookable']
for col in bool_cols:
    df[col] = df[col].map({'t': 1, 'f': 0})


# Handle neighbourhood and zipcode
df['neighbourhood'] = df['neighbourhood'].fillna('unknown')
df['zipcode'] = df['zipcode'].fillna('unknown')

output_path=(r"C:/Users/ACER/Desktop/DataAnalyticsTasks/Cleaned_Airbnb.xlsx")
df.to_excel(output_path , index=False)
print(f"\nCleaned dataset saved at: {output_path}")


#EDA
# Basic info
print(df.info())
print(df.describe())

# Distribution of target variable
sns.histplot(df['log_price'], kde=True)
plt.title("Distribution of Log Price")
plt.xlabel("Log Price")
plt.ylabel("Frequency")
plt.show()

# Select numeric columns only
numeric_df = df.select_dtypes(include='number')

# Boxplot by room type
plt.figure(figsize=(8, 5))
sns.boxplot(x='room_type', y='log_price', data=df)
plt.title("Log Price by Room Type")
plt.show()

# Boxplot by property type (top 10 types only for readability)
top_props = df['property_type'].value_counts().index[:10]
plt.figure(figsize=(10, 5))
sns.boxplot(x='property_type', y='log_price', data=df[df['property_type'].isin(top_props)])
plt.title("Log Price by Top 10 Property Types")
plt.xticks(rotation=45)
plt.show()

# Boxplot by city
plt.figure(figsize=(10, 5))
sns.boxplot(x='city', y='log_price', data=df)
plt.title("Log Price by City")
plt.xticks(rotation=45)
plt.show()


# Correlation matrix
corr_matrix = numeric_df.corr()

# Plotting heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap of Numerical Features")
plt.show()

#regression models
#converting log price to actual price (we need it to predict meaningful rental prices)
#create new coln (y)
df['price']= np.exp(df['log_price'])

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

# Target variable
y = df['log_price']

# Feature columns

x = df[['room_type', 'city', 'review_season', 'number_of_reviews', 'review_scores_rating',
        'accommodates', 'bathrooms', 'bedrooms', 'beds']]


# Split data
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Categorical and numeric columns
cat_features = ['room_type', 'city', 'review_season']

num_features = ['number_of_reviews', 'review_scores_rating',
                'accommodates', 'bathrooms', 'bedrooms', 'beds']

# Column transformer
preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
], remainder='passthrough')

# 🔹 Linear Regression pipeline
lr_pipeline = Pipeline([
    ('preprocess', preprocessor),
    ('model', LinearRegression())
])

# 🔹 Random Forest pipeline
rf_pipeline = Pipeline([
    ('preprocess', preprocessor),
    ('model', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Fit models
lr_pipeline.fit(X_train, y_train)
rf_pipeline.fit(X_train, y_train)

# Predict log_price
lr_preds_log = lr_pipeline.predict(X_test)
rf_preds_log = rf_pipeline.predict(X_test)

# Convert to actual price
import numpy as np
lr_preds_price = np.exp(lr_preds_log)
rf_preds_price = np.exp(rf_preds_log)
y_test_price = np.exp(y_test)

# Evaluation
print("Linear Regression:")
print("R² Score:", r2_score(y_test, lr_preds_log))
print("RMSE:", np.sqrt(mean_squared_error(y_test, lr_preds_log)))

print("\n Random Forest:")
print("R² Score:", r2_score(y_test, rf_preds_log))
print("RMSE:", np.sqrt(mean_squared_error(y_test, rf_preds_log)))



