import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

#read house_predictin_data_set
def load_and_explore_data(filepath):
    """Load dataset"""
    try:
        df = pd.read_csv(filepath, sep="\s+", header=None)
        print("=== Dataset Overview ===")
        print(f"\nDataset shape: {df.shape}")
        print("\nFirst 5 rows:")
        print(df.head())
        
        print("\nDataset info:")
        print(df.info())
        
        return df
    
    except FileNotFoundError:
        print(f"Error '{filepath}' not found.")
        return None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def evaluate_model(model, X_train, X_test, y_train, y_test):

    #Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    #metrics
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
     
    mse = mean_squared_error(y_test, y_test_pred)

    print("\n=== Evaluation ===")
    print(f"Train R²: {train_r2:.4f}")
    print(f"Test  R²: {test_r2:.4f}")
    print(f"Test  MSE: {mse:.4f}")

    #Check for overfitting
    r2_diff = train_r2 - test_r2
    if r2_diff > 0.1:
        print(f"\noverfitting (R² difference: {r2_diff:.4f})")
    elif test_r2 > train_r2:
        print("\ntest data")
    else:
        print(f"\nGood performance (R² difference: {r2_diff:.4f})")
    
    return y_test_pred, train_r2, test_r2, mse

def main():
    #Load and explore data
    df = load_and_explore_data("house_prediction_data_set.csv")
    if df is None:
        return
    
    #Prepare
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    
    print(f"\nFeatures shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    
    #Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42,
        shuffle=True
    )
    
    print(f"\nTraining set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    #Train
    print("\n=== Training Linear Regression Model ===")
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    #pos and neg
    print(f"\npos: {model.coef_}")
    print(f"\nneg: {model.intercept_:.4f}")
    
    #Evaluate
    y_pred = evaluate_model(model, X_train, X_test, y_train, y_test)
    
    return model, X_train, X_test, y_train, y_test, y_pred

# Run the analysis
if __name__ == "__main__":
    results = main()
