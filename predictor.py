import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os
import glob

class RealEstatePricePredictor:
    def __init__(self, data_file=None):
        """
        Initialize price prediction model
        Uses machine learning to predict property values
        """
        # Load data
        if data_file:
            self.data = pd.read_csv(data_file)
        else:
            # Find the most recent data file
            data_files = glob.glob('data/toronto_properties_*.csv')
            if data_files:
                latest_file = max(data_files, key=os.path.getctime)
                self.data = pd.read_csv(latest_file)
                print(f"Loaded data from: {latest_file}")
            else:
                raise FileNotFoundError("No property data files found. Run data_collector.py first.")
        
        self.model = None
        self.feature_columns = []
        self.label_encoders = {}
        
        # Create models directory
        if not os.path.exists('models'):
            os.makedirs('models')
        
        print(f"Initializing price predictor with {len(self.data)} properties")
    
    def prepare_features(self):
        """
        Prepare and engineer features for machine learning
        Convert categorical data to numeric and handle missing values
        """
        print("Preparing features for machine learning...")
        
        # Create a copy of data for feature engineering
        df = self.data.copy()
        
        # Calculate property age
        current_year = 2024
        if 'year_built' in df.columns:
            df['property_age'] = current_year - df['year_built']
        
        # Calculate price per square foot (useful feature)
        if 'square_feet' in df.columns:
            df['price_per_sqft'] = df['assessed_value'] / df['square_feet']
        
        # Encode categorical variables
        categorical_columns = ['neighborhood', 'property_type']
        for col in categorical_columns:
            if col in df.columns:
                le = LabelEncoder()
                df[col + '_encoded'] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
        
        # Select features for the model
        potential_features = [
            'bedrooms', 'bathrooms', 'square_feet', 'lot_size', 
            'property_age', 'neighborhood_encoded', 'property_type_encoded'
        ]
        
        # Only use features that exist in our data
        self.feature_columns = [col for col in potential_features if col in df.columns]
        
        print(f"Using features: {self.feature_columns}")
        
        # Prepare final dataset
        X = df[self.feature_columns].fillna(df[self.feature_columns].median())
        y = df['assessed_value']
        
        return X, y
    
    def train_model(self):
        """
        Train machine learning model to predict property prices
        Uses Random Forest for robust predictions
        """
        print("Training price prediction model...")
        
        # Prepare features
        X, y = self.prepare_features()
        
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train Random Forest model (good for real estate prediction)
        self.model = RandomForestRegressor(
            n_estimators=100,      # Number of trees
            max_depth=10,          # Prevent overfitting
            min_samples_split=5,   # Minimum samples to split
            random_state=42
        )
        
        self.model.fit(X_train, y_train)
        
        # Make predictions on test set
        y_pred = self.model.predict(X_test)
        
        # Calculate accuracy metrics
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"Model Performance:")
        print(f"Mean Absolute Error: ${mae:,.0f}")
        print(f"R² Score: {r2:.3f} ({r2*100:.1f}% of variance explained)")
        
        # Feature importance analysis
        feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nFeature Importance (what drives property prices):")
        for _, row in feature_importance.iterrows():
            print(f"• {row['feature']}: {row['importance']:.3f}")
        
        # Save the trained model
        model_filename = 'models/toronto_price_predictor.pkl'
        joblib.dump({
            'model': self.model,
            'feature_columns': self.feature_columns,
            'label_encoders': self.label_encoders
        }, model_filename)
        
        print(f"\nModel saved to: {model_filename}")
        return self.model
    
    def predict_price(self, property_features):
        """
        Predict price for a new property
        Takes property characteristics and returns estimated value
        """
        if self.model is None:
            print("Model not trained yet. Training now...")
            self.train_model()
        
        # Convert input to DataFrame
        if isinstance(property_features, dict):
            df_input = pd.DataFrame([property_features])
        else:
            df_input = property_features.copy()
        
        # Prepare features same way as training data
        current_year = 2024
        if 'year_built' in df_input.columns:
            df_input['property_age'] = current_year - df_input['year_built']
        
        # Encode categorical variables
        for col, encoder in self.label_encoders.items():
            if col in df_input.columns:
                df_input[col + '_encoded'] = encoder.transform(df_input[col].astype(str))
        
        # Select only the features used in training
        X_pred = df_input[self.feature_columns].fillna(0)
        
        # Make prediction
        predicted_price = self.model.predict(X_pred)[0]
        
        return predicted_price
    
    def predict_multiple_scenarios(self):
        """
        Test the model with different property scenarios
        Shows how different features affect price predictions
        """
        print("\n" + "="*60)
        print("PRICE PREDICTION SCENARIOS")
        print("="*60)
        
        if self.model is None:
            self.train_model()
        
        # Define test scenarios
        scenarios = [
            {
                'name': '2BR Downtown Condo',
                'bedrooms': 2,
                'bathrooms': 2,
                'square_feet': 800,
                'lot_size': 0,
                'year_built': 2015,
                'neighborhood': 'Downtown',
                'property_type': 'Condo'
            },
            {
                'name': '3BR Suburban House',
                'bedrooms': 3,
                'bathrooms': 2,
                'square_feet': 1500,
                'lot_size': 5000,
                'year_built': 1995,
                'neighborhood': 'Scarborough',
                'property_type': 'Detached'
            },
            {
                'name': '4BR Luxury Home',
                'bedrooms': 4,
                'bathrooms': 3,
                'square_feet': 2500,
                'lot_size': 8000,
                'year_built': 2010,
                'neighborhood': 'North York',
                'property_type': 'Detached'
            },
            {
                'name': '1BR Investment Condo',
                'bedrooms': 1,
                'bathrooms': 1,
                'square_feet': 600,
                'lot_size': 0,
                'year_built': 2020,
                'neighborhood': 'Liberty Village',
                'property_type': 'Condo'
            }
        ]
        
        print("Property Type Predictions:")
        print("-" * 60)
        
        for scenario in scenarios:
            name = scenario.pop('name')
            predicted_price = self.predict_price(scenario)
            
            print(f"\n{name}:")
            print(f"  Predicted Value: ${predicted_price:,.0f}")
            print(f"  Details: {scenario['bedrooms']}BR/{scenario['bathrooms']}BA, "
                  f"{scenario['square_feet']} sqft, {scenario['neighborhood']}")
        
        return scenarios
    
    def compare_neighborhoods(self):
        """
        Compare how neighborhood affects price for identical properties
        Useful for investment decisions
        """
        print("\n" + "="*60)
        print("NEIGHBORHOOD PRICE COMPARISON")
        print("="*60)
        
        if self.model is None:
            self.train_model()
        
        # Standard property profile
        base_property = {
            'bedrooms': 2,
            'bathrooms': 2,
            'square_feet': 1000,
            'lot_size': 3000,
            'year_built': 2000,
            'property_type': 'Townhouse'
        }
        
        neighborhoods = self.data['neighborhood'].unique()
        neighborhood_prices = []
        
        print("Same property (2BR/2BA Townhouse, 1000 sqft) in different neighborhoods:")
        print("-" * 60)
        
        for neighborhood in neighborhoods:
            test_property = base_property.copy()
            test_property['neighborhood'] = neighborhood
            
            predicted_price = self.predict_price(test_property)
            neighborhood_prices.append((neighborhood, predicted_price))
        
        # Sort by price
        neighborhood_prices.sort(key=lambda x: x[1], reverse=True)
        
        for neighborhood, price in neighborhood_prices:
            print(f"{neighborhood:<15}: ${price:>10,.0f}")
        
        # Calculate price spread
        highest = neighborhood_prices[0][1]
        lowest = neighborhood_prices[-1][1]
        spread = highest - lowest
        
        print(f"\nPrice Spread: ${spread:,.0f} ({spread/lowest*100:.1f}% difference)")
        print(f"Most Expensive: {neighborhood_prices[0][0]}")
        print(f"Most Affordable: {neighborhood_prices[-1][0]}")
        
        return neighborhood_prices
    
    def generate_prediction_report(self):
        """
        Generate comprehensive price prediction analysis
        Shows model performance and practical predictions
        """
        print("\n" + "#"*70)
        print("TORONTO REAL ESTATE PRICE PREDICTION REPORT")
        print("#"*70)
        
        # Train model if not already trained
        if self.model is None:
            self.train_model()
        
        # Run prediction scenarios
        scenarios = self.predict_multiple_scenarios()
        neighborhood_comparison = self.compare_neighborhoods()
        
        print("\n" + "="*60)
        print("PREDICTION MODEL INSIGHTS")
        print("="*60)
        
        print("✓ Machine learning model trained on Toronto real estate data")
        print("✓ Can predict property values based on location, size, and features")
        print("✓ Useful for investment analysis and market valuation")
        print("✓ Model accounts for neighborhood premiums and property characteristics")
        
        return {
            'model_performance': self.model,
            'scenarios': scenarios,
            'neighborhood_comparison': neighborhood_comparison
        }

# Test the predictor if running this file directly
if __name__ == "__main__":
    try:
        predictor = RealEstatePricePredictor()
        report = predictor.generate_prediction_report()
        
        print("\n" + "="*60)
        print("PRICE PREDICTION SYSTEM READY!")
        print("="*60)
        print("You can now predict property values for any Toronto property.")
        print("The model is saved and ready for use in your analysis pipeline.")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run 'python data_collector.py' first to collect data.")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")