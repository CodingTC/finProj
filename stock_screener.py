# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import yfinance as yf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from openai import OpenAI
import matplotlib.pyplot as plt
import seaborn as sns


client = OpenAI(api_key="YOUR_API_KEY") # Replace with your key


screening_criteria = {
    "net_profit_margin_min": 0.1, #Net Profit Margin >10%
    "dividend_yield_min": 0.002, #Dividend Yield >0.2%
    "eps_growth_min": 0.08, # EPS growth > 8%
    "market_cap_min": 10e9 # Market cap > $10 billion
}

tickers = ['NVDA', 'INTC', 'GOOGL', 'AMZN', 'META', 'MSFT', 'ORCL', 'AAPL', 'TSLA', 'T']

# Create empty list to store results
data = []
for ticker in tickers:
     stock = yf.Ticker(ticker)
     info = stock.info
     
     data.append({
     "ticker": ticker,
     "net_margin_profit": info.get("profitMargins"),
     "dividend_yield": info.get("dividendYield"),
     "eps_growth": info.get("earningsQuarterlyGrowth"),
     "market_cap": info.get("marketCap"),
     })
     

# Store in a dataframe
df = pd.DataFrame(data)

df["dividend_yield"] = df["dividend_yield"] / 100

# Remove rows with missing data in key screening metrics
df.dropna(subset=[
    "net_margin_profit",
    "dividend_yield",
    "eps_growth",
    "market_cap",
    ], inplace=True)

# Ensure numeric values are valid
df = df[df["net_margin_profit"] > 0]
df = df[df["eps_growth"] > 0]
df = df[df["market_cap"] > 0]

# filter the dataframe using our screening criteria
df = df[
    (df["net_margin_profit"] >= screening_criteria["net_profit_margin_min"]) &
    (df["dividend_yield"] >= screening_criteria["dividend_yield_min"]) &
    (df["eps_growth"] >= screening_criteria["eps_growth_min"]) &
    (df["market_cap"] >= screening_criteria["market_cap_min"])
]

# set scaler to min max normalization - rescales numbers into 0-1 range
scaler = MinMaxScaler()

# apply normalization formula to every row and add a new column for it
# to the data frame
df['profit_margin_score'] = scaler.fit_transform(df[['net_margin_profit']])
df['dividend_score'] = scaler.fit_transform(df[['dividend_yield']])
df['eps_score'] = scaler.fit_transform(df[['eps_growth']])
df['marketcap_score'] = scaler.fit_transform(df[['market_cap']])

# add composite score column for each row based on our chosen weights
df['composite_score'] = (
    0.30 * df['eps_score'] +
    0.25 * df['profit_margin_score'] +
    0.25 * df['dividend_score'] +
    0.20 * df['marketcap_score']
)

# sort top 5 based on composite score
top_picks = df.sort_values("composite_score", ascending=False).head(5)

# print out the stock with its composite score
print(top_picks[['ticker','composite_score']])



summaries = []

#let chat gpt do its thing and summarize based off our values
def generate_prompt(row):
    return f"""
Summarize this investment opportunity in 3-4 sentences for a finance-savvy investor.

Ticker: {row['ticker']}
Net Profit Margin: {row['net_margin_profit']:.1%}
Dividend Yield: {row['dividend_yield']:.2%}
EPS Growth: {row['eps_growth']:.1%}
Market Cap: {row['market_cap']:,}
Composite Score: {row['composite_score']:.2f}
"""
for _, row in top_picks.iterrows():
    try:
        prompt = generate_prompt(row)

        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )

        summaries.append(response.choices[0].message.content.strip())

    except Exception as e:
        print("ERROR for", row["ticker"], ":", e)
        summaries.append("Summary failed.")

# Add summaries column to dataframe
top_picks['summary'] = summaries



# Make a bar chart of our top 5 composite scores
plt.figure(figsize=(10,6))

top_sorted = top_picks.sort_values("composite_score")

# Make a bar chart for the composite scores
plt.barh(top_sorted['ticker'], top_sorted['composite_score'], color='skyblue')

plt.xlabel("Composite Score")
plt.title("Top Stock Picks")

plt.tight_layout()
plt.savefig("composite_scores.png")
plt.show()


# create scatter plot to see which companies have both strong
# profit margins and strong earnings growth
# EPS vs Net Profit Margin
plt.figure(figsize=(8,6))

plt.scatter(df['net_margin_profit'], df['eps_growth'], alpha=0.7)

for _, row in top_picks.iterrows():
    plt.annotate(row['ticker'], (row['net_margin_profit'], row['eps_growth']))

plt.xlabel("Net Profit Margin")
plt.ylabel("EPS Growth")
plt.title("EPS Growth vs Net Profit Margin")

plt.grid(True)
plt.tight_layout()

plt.savefig("growth_vs_margin.png")
plt.show()



top_picks.to_csv("top_picks_with_summaries.csv", index=False)


