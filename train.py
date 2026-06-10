import joblib
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge

def train_model():
    """
    TRAINING PIPELINE: DIABETES PROGRESSION PREDICTOR
    
    DATASET OVERVIEW:
    - Source: National Institute of Diabetes and Digestive and Kidney Diseases.
    - Samples: 442 diabetic patients.
    - Inputs (10 Features): Age, Sex, BMI, BP, and 6 blood serum measurements (s1-s6).
      *Note: Features are pre-normalized and scaled by scikit-learn (mean=0, variance=1).
    - Target: A continuous quantitative index (25 to 350) measuring disease 
      progression one year after the baseline measurements were taken.
    """
    
    print("📦 Step 1: Loading historical patient data...")
    diabetes = load_diabetes()
    X, y = diabetes.data, diabetes.target
    
    print("✂️ Step 2: Splitting data into Train (80%) and Test (20%) sets...")
    # The test set is isolated in a 'vault' to evaluate true model performance later.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print("🏋️‍♂️ Step 3: Training the Ridge Regression model...")
    # Ridge regression applies L2 regularization to prevent the model from 
    # over-relying on any single health feature, keeping the weights stable.
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    
    print("📊 Step 4: Evaluating model performance on unseen data...")
    # The R2 score calculates what percentage of the variance in disease 
    # progression our mathematical formula successfully explains.
    score = model.score(X_test, y_test)
    print(f"🎯 R2 Score on Test Set: {score:.4f}")
    
    print("💾 Step 5: Serializing the trained model artifact...")
    # We freeze the trained mathematical weights into a portable binary file 
    # so our API container can load it and serve live inferences later.
    joblib.dump(model, "model.joblib")
    print("🚀 Model successfully saved as model.joblib!")

if __name__ == "__main__":
    train_model()