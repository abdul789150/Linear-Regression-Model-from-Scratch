import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from LinearRegressionModel import LinearRegressionModel
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# Reading DataSet
dataFrame = pd.read_csv("auto-mpg.csv")

# Spliting Features and labels
features = dataFrame.drop(["mpg", "car"], axis=1)
labels = dataFrame["mpg"]

# features info
print("Features Information: ")
print(features.info())
print("\n")


# converting into numpy array
features = features.values

# Training and Testing Subsets
x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.20)


# training and testing split information
print("Number of Training data points: ", str(x_train.shape[0]))
print("Number of Testing data points: ", str(x_test.shape[0]))
print("\n")

# Adding value "1" as bias in features
x_train = np.append(np.ones((x_train.shape[0], 1)), x_train, axis=1) 


# Initializing LinearRegressionModel
model = LinearRegressionModel()

# Initializing weights
weights = model.initialize_weights(x_train)
print("Initialized Weights: ", str(weights))
print("\n")

# Training Dataset
mean_squared_error_history, final_optimized_weights = model.train(x_train, y_train, epochs=70000)

# optimized Weights
print("Optimized Weights: ", str(final_optimized_weights))

# prediction on testing set
x_test = np.append(np.ones((x_test.shape[0], 1)), x_test, axis=1) 
y_pred = model.predict(x_test)

testing_score = r2_score(y_test, y_pred)
print("\n")
print("R2 Score for Testing data: ", str(testing_score))
print("==============================================================================================//")



# Plotting mean squared error history
plt.plot(mean_squared_error_history)
plt.xlabel("No. of Epochs")
plt.ylabel("Mean Squared Error")
plt.title("Optimizing Cost using Gradient Descent")
plt.show()
