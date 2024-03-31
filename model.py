import numpy as np


np.random.seed(0)  

num_passengers = 1000
age = np.random.randint(1, 91, size=num_passengers)
sex = np.random.choice(['male', 'female'], size=num_passengers)
pclass = np.random.randint(1, 4, size=num_passengers)


survived = np.random.randint(0, 2, size=num_passengers)

sex_encoded = np.where(sex == 'male', 1, 0)


pclass_encoded = np.zeros((num_passengers, 3))
for i in range(num_passengers):
    pclass_encoded[i, pclass[i] - 1] = 1


X = np.column_stack((age, sex_encoded, pclass_encoded))

X = np.column_stack((np.ones(num_passengers), X))


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


num_features = X.shape[1]
theta = np.zeros(num_features)


def cost_function(X, y, theta):
    m = len(y)
    h = sigmoid(X.dot(theta))
    cost = -(1/m) * np.sum(y * np.log(h) + (1-y) * np.log(1 - h))
    return cost


def gradient_descent(X, y, theta, alpha, num_iterations):
    m = len(y)
    cost_history = []
    for _ in range(num_iterations):
        h = sigmoid(X.dot(theta))
        gradient = (1/m) * X.T.dot(h - y)
        theta -= alpha * gradient
        cost_history.append(cost_function(X, y, theta))
    return theta, cost_history


alpha = 0.01
num_iterations = 1000
theta, cost_history = gradient_descent(X, survived, theta, alpha, num_iterations)


def predict(X, theta):
    return np.round(sigmoid(X.dot(theta)))

predictions = predict(X, theta)


accuracy = np.mean(predictions == survived)
print("Accuracy:", accuracy)
