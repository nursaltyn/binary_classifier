import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import f1_score

def plot_conf_matrix(conf_matrix, y_test, descr):
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False,
    xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix {descr}, test set')
    plt.show()
    
    
    
def eval_run(my_data, model='ridge'):
    predictors = list(my_data.columns)
    predictors.remove("ID")
    predictors.remove("Type_y")

    X = my_data[predictors]
    y = my_data['Type_y']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    if model == 'ridge':
        ridge_model = RidgeClassifier(alpha=1)  
        ridge_model.fit(X_train, y_train)
        y_pred = ridge_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy:.4f}")
        f1 = f1_score(y_test, y_pred, average='weighted')
        print(f"F1_score: {f1:.4f}")
        conf_matrix = confusion_matrix(y_test, y_pred)
        class_report = classification_report(y_test, y_pred)
        plot_conf_matrix(conf_matrix, y_test, "RidgeClassifier")
        
    elif model == 'logistic':
        logistic_model = LogisticRegression(
        penalty='l2')

        logistic_model.fit(X_train, y_train)
        y_pred = logistic_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy:.4f}")
        f1 = f1_score(y_test, y_pred, average='weighted')
        print(f"F1_score: {f1:.4f}")
        conf_matrix = confusion_matrix(y_test, y_pred)
        plot_conf_matrix(conf_matrix, y_test, "LogisticRegression")
        
    elif model == 'random_forest':
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        print(f'Accuracy: {accuracy:.4f}')
        f1 = f1_score(y_test, y_pred, average='weighted')
        print(f"F1_score: {f1:.4f}")
        plot_conf_matrix(conf_matrix, y_test, "RandomForests")

        