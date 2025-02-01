import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# read the dataset
df = pd.read_csv("car_sales_dataset.csv")

# extract first 10 rows
print(df.head(10))

# extract last 10 rows
print(df.tail(10))

# extract columns
print(df.columns)
"""
['Car_ID', 'Make', 'Model', 'Year', 'Price', 'Mileage', 'Fuel_Type',
       'Transmission', 'Engine_Size', 'Horsepower', 'Torque', 'Drivetrain',
       'Color', 'Number_of_Doors', 'Number_of_Seats', 'Owner_Type', 'Location',
       'Seller_Type', 'Condition', 'Listing_Date']
"""

# overview of dataset

print(df.info())  #dtypes: float64(1), int64(8), object(11)
print("Check null values ",df.isnull().sum())
print("Check the duplicate data",df["Car_ID"].duplicated().sum())
print("Extract statistical values ",df.describe())


# check datatype and modify
df['Listing_Date']=pd.to_datetime(df['Listing_Date'],dayfirst=True,errors='coerce')

# print(df.info())

# Create new Feature
print("-------------------------------------------------------------------")
#Total number of unique car makes
print("Unique Car Makes:\n", df["Make"].nunique())
print("-------------------------------------------------------------------")

#Most expensive car in the dataset
print("Most Expensive Car:\n", df.loc[df["Price"].idxmax()])
print("-------------------------------------------------------------------")

#Least expensive car in the dataset
print("Least Expensive Car:\n", df.loc[df["Price"].idxmin()])
print("-------------------------------------------------------------------")

#Count of cars by fuel type
print("Cars by Fuel Type:\n", df["Fuel_Type"].value_counts())
print("-------------------------------------------------------------------")

#Most common car color
print("Most Common Color:          ", df["Color"].mode()[0])
print("-------------------------------------------------------------------")

#Average horsepower per car brand
print("Average Horsepower by Make:\n", df.groupby("Make")["Horsepower"].mean())
print("-------------------------------------------------------------------")

#Cars available per location
print("Cars per Location:\n", df["Location"].value_counts())
print("-------------------------------------------------------------------")

#Count of automatic vs manual transmissions
print("Transmission Count:\n", df["Transmission"].value_counts())
print("-------------------------------------------------------------------")

#Number of cars per drivetrain type
print("Drivetrain Distribution:\n", df["Drivetrain"].value_counts())
print("-------------------------------------------------------------------")

#Average price of cars by owner type
print("Average Price by Owner Type:\n", df.groupby("Owner_Type")["Price"].mean())
print("-------------------------------------------------------------------")

#Top 5 cheapest cars
print("Top 5 Cheapest Cars:\n", df.nsmallest(5, "Price"))
print("-------------------------------------------------------------------")

#Top 5 most expensive cars
print("Top 5 Most Expensive Cars:\n", df.nlargest(5, "Price"))
print("-------------------------------------------------------------------")

#Cars listed after a specific date
date_filter = "2024-01-01"
print(f"Cars listed after {date_filter}:\n", df[df["Listing_Date"] > date_filter])
print("-------------------------------------------------------------------")

#Most powerful car in terms of horsepower
print("Most Powerful Car:\n", df.loc[df["Horsepower"].idxmax()])
print("-------------------------------------------------------------------")


# data Visualization
#Price Distribution
plt.figure(figsize=(8,5))
sns.histplot(df["Price"], bins=30, kde=True)
plt.title("Price Distribution of Cars")
plt.show()

#Transmission Type Distribution
plt.figure(figsize=(8,5))
sns.countplot(x="Transmission", data=df, palette="pastel")
plt.title("Transmission Type Distribution")
plt.show()

#Car Listings per Location
plt.figure(figsize=(8,5))
df["Location"].value_counts().plot(kind="bar", color="skyblue")
plt.title("Number of Car Listings per Location")
plt.xticks(rotation=45)
plt.show()

#Average Car Price by Make
plt.figure(figsize=(8,5))
df.groupby("Make")["Price"].mean().sort_values(ascending=False).plot(kind="bar", color="orange")
plt.title("Average Car Price by Make")
plt.xticks(rotation=45)
plt.show()

#Horsepower vs Price
plt.figure(figsize=(8,5))
sns.scatterplot(x=df["Horsepower"], y=df["Price"], alpha=0.7)
plt.title("Horsepower vs Price")
plt.show()

# Car Listings by Year
plt.figure(figsize=(8,5))
df["Year"].value_counts().sort_index().plot(kind="bar", color="green")
plt.title("Car Listings by Year")
plt.xticks(rotation=45)
plt.show()

#Mileage Distribution
plt.figure(figsize=(8,5))
sns.histplot(df["Mileage"], bins=30, kde=True, color="purple")
plt.title("Mileage Distribution of Cars")
plt.show()