# deploy-ml-model-flask
 deploy your ml model using flask

 
HOW TO DEAL WITH THE SETUP ISSUES : basics 

1) do not setup the iri.pkl file inside your virtual env or the local repo . ( Flask apps often run from a different directory than expected ) .

2) try to locate the right path of the install libraries , apt , python interpretors etc . to do so

   Open System Properties:

Press Win + R, type sysdm.cpl, and hit Enter.

Go to the Advanced tab and click Environment Variables.

Under System Variables, find Path, select it, and click Edit.

Click New and add the path:

```
C:\Users\lenovo\AppData\Roaming\Python\Python313\Scripts
```

Now, you should be able to run flask from anywhere.

3) to fix the iri.pkl issue create a new fitted model .

This script trains a RandomForestClassifier on the Iris dataset and saves it as `iri.pkl`.

## Python Code

```python
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

with open("iri.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model trained and saved as iri.pkl")

4)run app.py only 
