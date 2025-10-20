import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics


# 1)
customers = pd.read_csv('Ecommerce Customers')
print(customers.head())
print(customers.info())
print(customers.describe())

# 2)
sns.jointplot(x='Time on Website',y='Yearly Amount Spent',data=customers)
plt.show()

# 3)
sns.jointplot(x='Time on App',y='Yearly Amount Spent',data=customers)
plt.show()

# 4)
sns.jointplot(x='Time on App',y='Length of Membership',data=customers,kind='hex')
plt.show()

# 5)
sns.pairplot(customers)
plt.show()

# 6)
sns.lmplot(x='Yearly Amount Spent',y='Length of Membership',data=customers)
plt.show()

# 7) Training and Testing Data
X = customers[['Avg. Session Length','Time on App','Time on Website','Length of Membership']]
y = customers['Yearly Amount Spent']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=101)

# 8) Create an instance of a LinearRegression() model named lm.
lm = LinearRegression()

# 9)
lm.fit(X_train,y_train)

# 10) Print out the coefficients of the model
print(lm.coef_)

# 11) Predicting Test Data
predictions = lm.predict(X_test)

# 12)
plt.scatter(y_test,predictions)
plt.show()

# 13) Evaluating the Model
print('MAE: ',metrics.mean_absolute_error(y_test,predictions))
print('MSE: ',metrics.mean_squared_error(y_test,predictions))
print('RMSE: ',np.sqrt(metrics.mean_squared_error(y_test,predictions)))

# 14)
sns.displot((y_test-predictions),bins=50)
plt.show()

# 15)
cdf = pd.DataFrame(lm.coef_,['Avg. Session Length','Time on App','Time on Website','Length of Membership'],columns=['Coefficient'])
print(cdf.head())