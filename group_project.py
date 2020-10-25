
import pandas as pd
pathToCustomerData = "~/Downloads/bank/bank.csv"
customer_data = pd.read_csv(pathToCustomerData, sep=";")
print(customer_data)