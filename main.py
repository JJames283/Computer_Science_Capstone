import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Retrieve the cleaned movies dataset CSV file
DF = pd.read_csv('movies5years.csv')
DF.head()

# Split movie dataset into X and Y sets. X set is the independent variable (budget), and
# Y is the dependent variable (box office gross)
X = DF.iloc[:, :1]  # independent variable (budget)
Y = DF.iloc[:, 1:]  # dependent variable (box office gross)

# Split movie dataset into X and Y training and testing sets. Set test size to 0.2 (20%) and random state to 0.
X_training_set, X_testing_set, Y_training_set, Y_testing_set = train_test_split(X, Y, test_size=0.2, random_state=0)

# Set LinRegress to use Linear Regression model. LinRegress then used on X and Y training set.
LinRegress = LinearRegression()
LinRegress.fit(X_training_set.values, Y_training_set.values)

# Prediction training and testing for predicted value of Y_prediction_test and Y_prediction_train when given
# X value. In other words, what is the box office gross when given a film's budget.
Y_prediction_testing = LinRegress.predict(X_testing_set.values)
Y_prediction_training = LinRegress.predict(X_training_set.values)

# Blank space (for aesthetics)
print("\n")

# User is asked to enter a budget amount.
while True:
    try:
        print("Welcome to the Film Box Office Gross Prediction Tool.")
        print("Using linear regression, machine learning and historical data, this tool will predict a film's box "
              "office gross based on its budget.")
        print("Please note, the box office gross prediction may be negative if the budget is too low.")
        budget = input("Enter the film's budget in dollars, using whole numbers and no commas (example: 58000000):\n")
        budget = int(budget)
        break
    except Exception:
        # The user will receive an error message if they do not enter a valid number.
        print("Please enter a valid number.")

# Film box office gross prediction is generated based on the budget amount the user entered.
GrossPrediction = LinRegress.predict([[budget]])
# Prediction amount converted to int (whole numbers only), otherwise would have multiple decimal places.
UpdateToInt = int(GrossPrediction)
# updated prediction amount is printed
print(f"Box Office Gross Prediction: {UpdateToInt}")
