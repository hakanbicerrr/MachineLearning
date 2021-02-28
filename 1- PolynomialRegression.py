def answer_one():
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures
    import numpy as np
    from sklearn.model_selection import train_test_split

    np.random.seed(0)
    n = 15
    x = np.linspace(0, 10, n) + np.random.randn(n) / 5
    y = np.sin(x) + x / 6 + np.random.randn(n) / 10
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=0)

    test_data = np.linspace(0, 10, 100).reshape(-1, 1)
    data_pred = []
    for i in [1, 3, 6, 9]:
        poly = PolynomialFeatures(degree=i)
        x_poly = poly.fit_transform(X_train)
        model = LinearRegression()
        model.fit(x_poly, y_train)
        y_poly = poly.fit_transform(test_data)
        pred = model.predict(y_poly)
        data_pred.append(pred)
    result = np.array(data_pred).reshape(4, 100)
    return result

#For Visualizing
def plot_one(degree_predictions):
    import matplotlib.pyplot as plt
    %matplotlib notebook
    plt.figure(figsize=(10,5))
    plt.plot(X_train, y_train, 'o', label='training data', markersize=10)
    plt.plot(X_test, y_test, 'o', label='test data', markersize=10)
    for i,degree in enumerate([1,3,6,9]):
        plt.plot(np.linspace(0,10,100), degree_predictions[i], alpha=0.8, lw=2, label='degree={}'.format(degree))
    plt.ylim(-1,2.5)
    plt.legend(loc=4)

#plot_one(answer_one())