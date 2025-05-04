from sklearn.naive_bayes import GaussianNB 
import numpy as np 
 
X_train = np.array([[1, 0, 1], 
                    [0, 0, 0], 
                    [1, 1, 1], 
                    [0, 0, 0], 
                    [1, 0, 0], 
                    [0, 1, 0], 
                    [1, 1, 1]]) 
 
y_train = np.array([1, 0, 1, 0, 1, 0, 1]) 
X_test = np.array([[1, 0, 1], 
                   [0, 1, 0], 
                   [0, 0, 0],
                   [0, 1, 1]]) 
model = GaussianNB() 
model.fit(X_train, y_train) 
y_pred = model.predict(X_test) 
predicted_classes = ['Spam' if label == 1 else 'Not Spam' for label in 
y_pred] 
for i, prediction in enumerate(predicted_classes): 
    print(f"Email {i+8}: {prediction}") 

