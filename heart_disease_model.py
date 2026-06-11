
import pandas as pd
df = pd.read_csv('heart.csv')

print("Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nColumn names:", df.columns.tolist())
print("\nMissing values:")
print(df.isnull().sum())


from sklearn.preprocessing import LabelEncoder

df.fillna(df.median(numeric_only=True), inplace=True)

for col in df.select_dtypes(include='object').columns:
    df[col] = LabelEncoder().fit_transform(df[col].astype(str))

print("\nData cleaned successfully!")


from sklearn.model_selection import train_test_split

TARGET = 'target'
X = df.drop(columns=[TARGET])
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

print("\nTraining size:", X_train.shape)
print("Testing size:", X_test.shape)


from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)

dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

print("\nAll models trained successfully!")


from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

rf_acc = accuracy_score(y_test, rf_model.predict(X_test))*100
lr_acc = accuracy_score(y_test, lr_model.predict(X_test))*100
dt_acc = accuracy_score(y_test, dt_model.predict(X_test))*100

print("\n=== MODEL COMPARISON ===")
print(f"Random Forest       : {rf_acc:.2f}%")
print(f"Logistic Regression : {lr_acc:.2f}%")
print(f"Decision Tree       : {dt_acc:.2f}%")

best_model = rf_model
best_acc = rf_acc
best_name = "Random Forest"

if lr_acc > best_acc:
    best_model = lr_model
    best_acc = lr_acc
    best_name = "Logistic Regression"

if dt_acc > best_acc:
    best_model = dt_model
    best_acc = dt_acc
    best_name = "Decision Tree"

print(f"\nBest Model: {best_name} with {best_acc:.2f}% accuracy ✅")

y_pred = best_model.predict(X_test)

print("\nDetailed Report:")
print(classification_report(y_test, y_pred))


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(6,4))
sns.heatmap(confusion_matrix(y_test, y_pred),
            annot=True, fmt='d', cmap='Reds')
plt.title(f'Confusion Matrix - {best_name}')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.show()


feat_imp = pd.Series(best_model.feature_importances_,
                     index=X.columns).sort_values(ascending=False)
plt.figure(figsize=(8,5))
feat_imp[:10].plot(kind='bar', color='red')
plt.title('Top 10 Most Important Features')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.show()


print("\n=== TESTING WITH NEW PERSON DATA ===")





person1 = pd.DataFrame([[58, 0, 0, 100, 248, 0, 0, 122, 0, 1.0, 1, 0, 2]], 
                        columns=X.columns)


person2 = pd.DataFrame([[44, 1, 2, 120, 263, 0, 1, 173, 0, 0.0, 2, 0, 3]], 
                        columns=X.columns)


person3 = pd.DataFrame([[40, 1, 0, 110, 167, 0, 0, 114, 1, 2.0, 1, 0, 3]], 
                        columns=X.columns)


person4 = pd.DataFrame([[55, 1, 1, 130, 262, 0, 1, 155, 0, 0.0, 2, 0, 2]], 
                        columns=X.columns)

pred1 = best_model.predict(person1)
pred2 = best_model.predict(person2)
pred3 = best_model.predict(person3)
pred4 = best_model.predict(person4)

print(f"Person 1 (58yr female) : {'HEART DISEASE ❌' if pred1[0]==1 else 'NO HEART DISEASE ✅'}")
print(f"Person 2 (44yr male)   : {'HEART DISEASE ❌' if pred2[0]==1 else 'NO HEART DISEASE ✅'}")
print(f"Person 3 (40yr male)   : {'HEART DISEASE ❌' if pred3[0]==1 else 'NO HEART DISEASE ✅'}")
print(f"Person 4 (55yr male)   : {'HEART DISEASE ❌' if pred4[0]==1 else 'NO HEART DISEASE ✅'}")
print("\nVerifying from actual dataset:")
print(df[df['age']==44][['age','sex','cp','target']].head(3))
print(df[df['age']==58][['age','sex','cp','target']].head(3))
print(df[df['age']==40][['age','sex','cp','target']].head(3))
print(df[df['age']==55][['age','sex','cp','target']].head(3))
# ============ SAVE BEST MODEL ============
import joblib
joblib.dump(best_model, 'heart_disease_model.pkl')
print("\nModel saved!")