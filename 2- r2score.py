def answer_two():
    from sklearn.metrics.regression import r2_score
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
    r2_train = []
    r2_test = []
    for i in range(10):
        poly = PolynomialFeatures(degree=i)
        x_poly = poly.fit_transform(X_train)
        model = LinearRegression()
        model.fit(x_poly, y_train)
        y_pred_train = poly.fit_transform(X_train)
        y_pred_train = model.predict(y_pred_train)
        r2_train.append(r2_score(y_train, y_pred_train))
        y_pred_test = poly.transform(X_test)
        y_pred_test = model.predict(y_pred_test)
        r2_test.append(r2_score(y_test, y_pred_test))
    r2_train = np.array(r2_train)
    r2_test = np.array(r2_test)

    # Your code here

    return r2_train, r2_test