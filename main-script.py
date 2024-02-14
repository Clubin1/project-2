# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import seaborn as sns
import matplotlib.pyplot as plt

# %%
df = pd.read_csv('bitstampUSD_1-min_data_2012-01-01_to_2021-03-31.csv')
df

# %%
df.dropna(inplace=True)
df.head(20)

# %%
df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')
df['day_of_week'] = df['Timestamp'].dt.dayofweek
df['hour_of_day'] = df['Timestamp'].dt.hour

scaler = StandardScaler()
df[['Open', 'High', 'Low', 'Close', 'Volume_(BTC)', 'Volume_(Currency)', 'Weighted_Price']] = scaler.fit_transform(
    df[['Open', 'High', 'Low', 'Close', 'Volume_(BTC)', 'Volume_(Currency)', 'Weighted_Price']])

df['Price_Movement'] = np.where(df['Close'].shift(-1) > df['Close'], 'Up',
                                np.where(df['Close'].shift(-1) < df['Close'], 'Down', 'Stable'))

df = df.dropna()
df.head(20)

# %%
sampled_df = df.sample(n=1000)
sns.pairplot(sampled_df[['Open', 'High', 'Low', 'Close', 'Volume_(BTC)', 'Volume_(Currency)', 'Weighted_Price']])
plt.show()

sns.boxplot(x=sampled_df['Weighted_Price'])
plt.show()

sns.countplot(x='Price_Movement', data=sampled_df)
plt.show()
categorical_cols = ['Price_Movement']

df_categorical = df[categorical_cols].apply(lambda x: pd.factorize(x)[0])

df_combined = pd.concat([df.select_dtypes(include=['float64']), df_categorical], axis=1)

correlation_matrix = df_combined.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()

plt.figure(figsize=(12, 6))
sns.lineplot(x='Timestamp', y='Weighted_Price', data=sampled_df)
plt.title('Time Series Plot of Weighted Price')
plt.show()

# %%
X = df[['Open', 'High', 'Low', 'Close', 'Volume_(BTC)', 'Volume_(Currency)', 'Weighted_Price', 'day_of_week', 'hour_of_day']]
y = df['Price_Movement']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %%
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features="sqrt",
    n_jobs=-1,
    random_state=42
)
model.fit(X_train, y_train)

# %%
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))


