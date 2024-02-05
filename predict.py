def predict_y(W,b,X, n_layers):
    input_len = X.shape[0]
    output_predict = np.zeros((input_len,))
    for i in range(input_len):
        h, z = feed_forward(X[i,:], W,b)
        y[i] = np.argmax(h[n_layers])
        return output_predict


y_pred = predict_y(W, b, x_test, 3)
print (y_pred)
