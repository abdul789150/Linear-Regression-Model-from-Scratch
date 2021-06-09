import numpy as np
from sklearn.metrics import r2_score


class LinearRegressionModel():

    weights = None

    # Initializing weights
    def initialize_weights(self, features, random=False):

        if random == True:
            self.weights = np.random.rand(features.shape[1])
        else:
            self.weights = np.zeros(features.shape[1])
        
        return self.weights

    # Cost function
    def mean_squared_error(self, actual, output):
        samples = len(actual)
        cost_sum = 0
        
        for y, y_hat in zip(actual, output):
            cost = (y - y_hat) ** 2
            cost_sum += cost
            
        error = cost_sum / (samples * 2)
        
        return error

    # Gradient Function #00
    def gradient_step(self, x, y, y_hat, total_samples, alpha=0.0000001):
        
        derivate_error = -(y - y_hat)
        derivate_error = x.dot(derivate_error)
        self.weights = self.weights - alpha/total_samples * derivate_error
        
        # return weights

    # Model training
    def train(self, features, labels, epochs, alpha=0.0000001, error_info=False):
    
        errors = []
        
        if error_info == False:
            print("Training the model for ", str(epochs)," epochs, Please Wait....")


        for e in range(epochs):

            # applying linear regression y_hat = w(i)*x(i) + ... + w(n) * x(n)
            predictions = []
            for data_point in features:
                y_hat = 0
                for w, x in zip(self.weights, data_point):
                    y_hat = y_hat + (w*x)

                predictions.append(y_hat)

            # Calculating Error
            error = self.mean_squared_error(actual=labels, output=predictions)
            errors.append(error)

            if error_info != False:
                print("Epoch ", str(e+1), "/", str(epochs))
                print("Mean Squared Error: ", str(error))
                print("==========================================================================================//")

            # Applying Gradient Descent
            for x, y, y_hat in zip(features, labels, predictions):
                self.gradient_step(x, y, y_hat, len(labels), alpha=alpha)
            
        final_score = r2_score(labels, predictions)

        print("Final R2 Score for Training data: ", str(final_score))
        print("==============================================================================================//")

        return errors, self.weights


    def predict(self, features):

        predictions = []
        for data_point in features:
            y_hat = 0
            for w, x in zip(self.weights, data_point):
                y_hat = y_hat + (w*x)

            predictions.append(y_hat)
            
        return predictions
