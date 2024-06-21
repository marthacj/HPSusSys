
#this file takes ot the cpu highest average for 24 core and 28 core machines from the csv file
import pandas as pd

# Load the provided Excel file
file_path = 'c:\\Users\\martha.calder-jones\\OneDrive - University College London\\UCL_comp_sci\\Summer Project\\1038-0610-0614-day.xlsx'
df = pd.read_excel(file_path, skiprows=4)

# Print the column names to inspect them
print("Column names in the dataframe:", df.columns)

# Extract relevant columns based on the exact column names from the inspection
df_relevant = df[["#Cores", "CPU\nHighest\navg", "\nCPU\nTotal Seconds > 80%"]]

# Ensure numeric columns are properly parsed
df_relevant["CPU\nHighest\navg"] = pd.to_numeric(df_relevant["CPU\nHighest\navg"], errors='coerce')
df_relevant["\nCPU\nTotal Seconds > 80%"] = pd.to_numeric(df_relevant["\nCPU\nTotal Seconds > 80%"], errors='coerce')

# Replace NaN with 0 for calculations
df_relevant.fillna(0, inplace=True)

# Filter data based on the number of cores
df_24_core = df_relevant[df_relevant["#Cores"] == 24]
print(df_24_core["CPU\nHighest\navg"])
df_28_core = df_relevant[df_relevant["#Cores"] == 28]

# Calculate the average CPU Highest avg for each group
average_cpu_highest_avg_24_core = df_24_core["CPU\nHighest\navg"].mean()
average_cpu_highest_avg_28_core = df_28_core["CPU\nHighest\navg"].mean()

# Print the calculated averages
print("Average CPU Highest avg for 24-core machines:", average_cpu_highest_avg_24_core)
print("Average CPU Highest avg for 28-core machines:", average_cpu_highest_avg_28_core)



# import pandas as pd

# # Load the provided Excel file
# file_path = 'c:\\Users\\martha.calder-jones\\OneDrive - University College London\\UCL_comp_sci\\Summer Project\\1038-0610-0614-day.xlsx'
# df = pd.read_excel(file_path, skiprows=4)

# total_observation_time = 216000

# # Print the column names to inspect them
# print("Column names in the dataframe:", df.columns)

# # Extract relevant columns based on the exact column names from the inspection
# df_relevant = df[["#Cores", "CPU\nHighest\navg", "\nCPU\nTotal Seconds > 80%"]]

# # Ensure numeric columns are properly parsed
# # df_relevant["\nCPU\nTotal Seconds > 80%"] = pd.to_numeric(df_relevant["\nCPU\nTotal Seconds > 80%"], errors='coerce')

# # Replace NaN with 0 for calculations
# df_relevant.fillna(0, inplace=True)


# df_relevant.loc[:, "Combined_Utilization"] = df_relevant["CPU\nHighest\navg"]

# # df_relevant["High_Utilization_Percentage"] =  df_relevant["\nCPU\nTotal Seconds > 80%"] / total_observation_time

# # Define weighting factors (these can be adjusted based on the importance)
# # avg_weight = 0.3
# # high_utilization_weight = 0.7

# # # Calculate the combined utilization metric
# # df_relevant["Combined_Utilization"] = (avg_weight * df_relevant["CPU\nHighest\navg"]) + (high_utilization_weight *  df_relevant["High_Utilization_Percentage"])

# # Calculate the average combined CPU utilization
# # average_combined_cpu_utilization = df["Combined_Utilization"].mean()

# # Filter data based on the number of cores
# df_24_core = df_relevant[df_relevant["#Cores"] == 24]
# df_28_core = df_relevant[df_relevant["#Cores"] == 28]

# # Calculate the average combined CPU utilization for each group
# average_combined_cpu_24_core = df_24_core["Combined_Utilization"].mean()
# average_combined_cpu_28_core = df_28_core["Combined_Utilization"].mean()

# # Print the calculated averages
# print("Average combined CPU utilization for 24-core machines:", average_combined_cpu_24_core)
# print("Average combined CPU utilization for 28-core machines:", average_combined_cpu_28_core)