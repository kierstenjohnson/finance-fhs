import pandas as pd
import numpy as np
import statsmodels.api as sm

# Monthly industry data
industry_monthly = pd.read_csv('5_industry_portfolios.CSV', skiprows=11, nrows=1183)
industry_monthly = industry_monthly.rename(columns={'Unnamed: 0': 'Date'})
industry_monthly = industry_monthly.rename(columns={'Hlth ': 'Hlth'})
industry_monthly = industry_monthly.dropna().reset_index(drop=True)
industry_monthly["Date"] = industry_monthly["Date"].astype(str).str[:6].astype(int)

# Monthly market data
market_monthly = pd.read_csv('F-F_Research_Data_Factors.CSV',skiprows=2, nrows=1182)
market_monthly = market_monthly.rename(columns={'Unnamed: 0': 'Date'})
market_monthly = market_monthly.rename(columns={'Hlth ': 'Hlth'})
market_monthly = market_monthly[["Date", "Mkt-RF"]]
market_monthly["Date"] = market_monthly["Date"].astype(str).str[:6].astype(int)

# Yearly industry data
industry_yearly = pd.read_csv('5_industry_portfolios.CSV', skiprows=2485, nrows=98)
industry_yearly = industry_yearly.rename(columns={'Unnamed: 0': 'Date'})
industry_yearly = industry_yearly.rename(columns={'Hlth ': 'Hlth'})
industry_yearly = industry_yearly.dropna().reset_index(drop=True)
industry_yearly["Date"] = industry_yearly["Date"].astype(str).str[:4].astype(int)

# Yearly market data
market_yearly = pd.read_csv('F-F_Research_Data_Factors.CSV',skiprows=1188, nrows=98)
market_yearly = market_yearly.rename(columns={'Unnamed: 0': 'Date'})
market_yearly = market_yearly.rename(columns={'Hlth ': 'Hlth'})
market_yearly = market_yearly[["Date", "Mkt-RF"]]
market_yearly["Date"] = market_yearly["Date"].astype(str).str[:4].astype(int)

monthly = pd.merge(industry_monthly, market_monthly)
yearly = pd.merge(industry_yearly, market_yearly)

def calculate_beta(df, industry_col):
    X = df["Mkt-RF"]
    X = sm.add_constant(X)
    y = df[industry_col]

    model = sm.OLS(y, X).fit()
    return model.params["Mkt-RF"]

betas = pd.DataFrame(columns=["Industry", "Monthly Beta", "Yearly Beta"])

for industry in ["Cnsmr", "Manuf", "HiTec", "Hlth", "Other"]:
    monthly_beta = calculate_beta(monthly, industry).round(4)
    yearly_beta = calculate_beta(yearly, industry).round(4)
    betas.loc[len(betas)] = [industry, monthly_beta, yearly_beta]

betas.to_csv("betas.csv", index=False)
print(betas)





