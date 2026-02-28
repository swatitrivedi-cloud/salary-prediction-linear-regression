import numpy  as np 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

X = np.array([ [1, 0, 0], [2, 0, 0], [3, 1, 0], [4, 1, 0], [5, 2, 0], [6, 2, 0], [3, 2, 1], [5, 3, 1], [7, 4, 1], [8, 5, 1], [10, 6, 1], [12, 7, 1] ])
y = np.array([35, 38, 45, 50, 55, 60, 55, 70, 85, 95, 110, 125])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

score = model.score(X_test, y_test)
print("Linear Regression R² Score:", score)

exp = float(input("Enter your number of expreince: "))
pubs = int(input("Enter number of publication: "))

phd_input = input("Do you have a PhD? (yes/no): ").strip().lower()   #strips remove extra space #lower convert text to lowercase
phd = 1 if phd_input == "yes" else 0

user_features = np.array([[exp, pubs, phd]])

pred_salary = model.predict(user_features)

print(f"Estimated Salary: ${pred_salary[0]:.0f}k") #round to 0 decimal places and format as a float
