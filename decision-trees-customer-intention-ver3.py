# Decision Tree Analysis of Customer Intention Data
# by: Joe Domaleski
# See: https://blog.marketingdatascience.ai

# Load libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pydotplus
from PIL import Image
from sklearn.metrics import confusion_matrix, classification_report

# 1. Load dataset
data = pd.read_csv("online_shoppers_intention.csv")

# 2. Convert 'Revenue' to numerical (0 = No Purchase, 1 = Purchase)
data['Revenue'] = data['Revenue'].astype('category').cat.codes  

# Convert categorical features to numeric using Label Encoding
label_encoders = {}
categorical_columns = ['Month', 'VisitorType', 'Weekend']
for col in categorical_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# 3. Scale numeric features
numeric_features = ['Administrative', 'Administrative_Duration', 'Informational',
                    'Informational_Duration', 'ProductRelated', 'ProductRelated_Duration',
                    'BounceRates', 'ExitRates', 'PageValues']
scaler = StandardScaler()
data[numeric_features] = scaler.fit_transform(data[numeric_features])

# 4. Split data into training (70%) and testing (30%) sets
np.random.seed(123)
trainData, testData = train_test_split(data, test_size=0.3, stratify=data['Revenue'])

# Define features and target
features = ['PageValues', 'BounceRates', 'ExitRates', 'ProductRelated', 'ProductRelated_Duration', 'Administrative', 'Month']
target = 'Revenue'

# 5. Train a **more balanced Decision Tree**
model = DecisionTreeClassifier(
    random_state=123, 
    max_depth=4,  # Increase depth slightly
    min_samples_split=100,  # Allow more splits
    min_samples_leaf=50,  # Reduce minimum leaf size
    ccp_alpha=0.005,  # Less aggressive pruning
    max_features=None,  # Consider all features
    class_weight={0.0: 1, 1.0: 2},  # Balance underrepresented class
    criterion='entropy'
)
model.fit(trainData[features], trainData[target])

# 6. Visualize the **improved tree**
dot_data = export_graphviz(model, feature_names=features, class_names=['No Purchase', 'Purchase'], filled=True)
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_png("decision_tree_improved.png")

# Show the tree image
img = Image.open("decision_tree_improved.png")
img.show()

# 7. Make predictions
predictions = model.predict(testData[features])

# 8. Evaluate accuracy
conf_matrix = pd.DataFrame(confusion_matrix(testData[target], predictions), 
                           index=['No Purchase', 'Purchase'], columns=['Predicted No Purchase', 'Predicted Purchase'])
print("Confusion Matrix:")
print(conf_matrix)

# Print classification report
print("\nClassification Report:")
print(classification_report(testData[target], predictions))

# 9. Display feature importance
feature_importance = pd.DataFrame({'Feature': features, 'Importance': model.feature_importances_})
feature_importance = feature_importance.sort_values(by='Importance', ascending=False)
print("\nFeature Importance:")
print(feature_importance)
