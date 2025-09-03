import pandas as pd

# Load the dataset
df = pd.read_csv("open_meteo_trichy_hourly_10yr.csv")

# Display original size
print(f"📄 Original rows: {len(df)}")

# Filter out rows where BOTH direct and diffuse radiation are zero
filtered_df = df[~((df["Direct Rad (W/m2)"] == 0) & (df["Diffuse Rad (W/m2)"] == 0))]

# Save the cleaned data
filtered_df.to_csv("open_meteo_trichy_hourly_10yr_cleaned.csv", index=False)

# Display new size
print(f"✅ Cleaned rows: {len(filtered_df)}")
print("🎉 Data saved to: open_meteo_trichy_hourly_10yr_cleaned.csv")
