import pandas as pd
import matplotlib.pyplot as plt

# Load OPU index data
oil_data = pd.read_csv('data/oil.csv')
oil_data['date'] = pd.to_datetime(oil_data['date'], format='%m/%d/%Y')
oil_data.set_index('date', inplace=True)

# Load oil price data
oil_price_data = pd.read_csv('data/oil_price.csv', skiprows=4)  # Skip metadata rows
oil_price_data.columns = ['Month', 'Oil_Price']  # Rename columns
oil_price_data['Month'] = pd.to_datetime(oil_price_data['Month'], format='%b %Y')  # Parse dates
oil_price_data.set_index('Month', inplace=True)

# Align the oil price data with the OPU index data
aligned_data = oil_data.join(oil_price_data, how='inner')

# Plot OPU Index and Oil Price
fig, ax1 = plt.subplots(figsize=(16, 8))

# Plot OPU Index on the primary y-axis
ax1.plot(aligned_data.index, aligned_data['OPU_index'], linewidth=2, label='OPU Index', color='blue')
ax1.set_xlabel('Date', fontsize=14)
ax1.set_ylabel('OPU Index', fontsize=14, color='blue')
ax1.tick_params(axis='y', labelcolor='blue')
ax1.set_title('Oil Price Uncertainty (OPU) Index and Oil Price Over Time', fontsize=16)
ax1.grid(True, alpha=0.3)

# Plot Oil Price on the secondary y-axis
ax2 = ax1.twinx()
ax2.plot(aligned_data.index, aligned_data['Oil_Price'], linewidth=2, label='Oil Price', color='orange')
ax2.set_ylabel('Oil Price ($)', fontsize=14, color='orange')
ax2.tick_params(axis='y', labelcolor='orange')

# Add annotations for major events
events = {
    '1973-10-15': 'OPEC Oil Embargo',
    '1979-01-15': 'Iranian Revolution',
    '1990-08-02': 'Gulf War',
    '2008-07-11': 'Oil Price Peak',
    '2014-11-01': 'Oil Price Collapse',
    '2020-03-09': 'COVID-19 Oil Crash'
}

for date, event in events.items():
    date_obj = pd.to_datetime(date)
    if date_obj >= aligned_data.index.min() and date_obj <= aligned_data.index.max():
        closest_date = aligned_data.index[aligned_data.index.get_indexer([date_obj], method='nearest')[0]]
        value = aligned_data.loc[closest_date, 'OPU_index']
        ax1.annotate(event, 
                     xy=(closest_date, value),
                     xytext=(closest_date, value + 50),
                     arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
                     fontsize=12)

# Add legends for both axes
fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9), fontsize=12)

plt.tight_layout()
plt.savefig('opu_index_over_time.png')
plt.close()