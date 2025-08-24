import pandas as pd

# Load in chunks
chunks = pd.read_csv('Global_Cybersecurity_Threats_2015-2024.csv', chunksize=10000)
df = pd.concat(chunks)

# Quick #overview
print(df.shape)
print(df.columns)
print(df.head())

df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df.dropna(subset=['date', 'threat_type'], inplace=True)

import seaborn as sns
import matplotlib.pyplot as plt

# Threats over time
df['year'] = df['date'].dt.year
sns.countplot(data=df, x='year', order=sorted(df['year'].unique()))
plt.title('Cybersecurity Threats Over Time')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
df['threat_severity'] = df['impact_score'] * df['frequency']
from sklearn.linear_model import LinearRegression

# Example: Predict number of threats per year
threats_per_year = df.groupby('year').size().reset_index(name='count')
X = threats_per_year[['year']]
y = threats_per_year['count']

model = LinearRegression()
model.fit(X, y)

# Predict next 5 years
future_years = pd.DataFrame({'year': range(2025, 2030)})
predictions = model.predict(future_years)
# streamlit_app.py
import streamlit as st
st.title("Global Cybersecurity Threats Dashboard")
st.line_chart(threats_per_year.set_index('year'))
selected_year = st.slider("Select Year", min_value=int(df['year'].min()), max_value=2024)
filtered_df = df[df['year'] == selected_year]
st.bar_chart(filtered_df['threat_type'].value_counts())
future_years['predicted_threats'] = predictions.astype(int)
st.write("Predicted Threats (2025–2029):")
st.dataframe(future_years)
severity_by_year = df.groupby('year')['threat_severity'].mean().reset_index()
st.line_chart(severity_by_year.set_index('year'))
st.write("Threat Type Distribution")
st.bar_chart(df['threat_type'].value_counts())