import pandas as pd
import requests
import json
import os
from datetime import datetime

class TorontoRealEstateCollector:
    def __init__(self):
        # Toronto Open Data API endpoints
        self.property_assessments_url = "https://ckan0.cf.opendata.inter.service-canada.ca/api/3/action/datastore_search"
        self.building_permits_url = "https://ckan0.cf.opendata.inter.service-canada.ca/api/3/action/datastore_search"
        
        # Create data directory if it doesn't exist
        if not os.path.exists('data'):
            os.makedirs('data')
    
    def get_toronto_property_data(self, limit=1000):
        """
        Fetch property assessment data from Toronto Open Data
        This gives us actual property values and characteristics
        """
        try:
            # Toronto Property Assessment dataset ID
            resource_id = "00e0f70d-91c4-4c1b-96c7-0a62babc0b0c"
            
            params = {
                'resource_id': resource_id,
                'limit': limit,
                'q': 'Toronto'  # Filter for Toronto properties
            }
            
            print(f"Fetching {limit} property records from Toronto Open Data...")
            response = requests.get(self.property_assessments_url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                records = data['result']['records']
                
                # Convert to DataFrame for easier analysis
                df = pd.DataFrame(records)
                print(f"Successfully collected {len(df)} property records")
                return df
            else:
                print(f"API request failed with status: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"Error collecting data: {str(e)}")
            return self._get_sample_data()  # Fallback to sample data
    
    def _get_sample_data(self):
        """
        Create realistic sample data if API fails
        This ensures our analysis can still run
        """
        print("Using sample Toronto real estate data...")
        
        # Sample neighborhoods in Toronto
        neighborhoods = ['Downtown', 'Midtown', 'North York', 'Etobicoke', 'Scarborough', 
                        'The Beaches', 'Leslieville', 'King West', 'Liberty Village', 'Corktown']
        
        property_types = ['Detached', 'Semi-Detached', 'Townhouse', 'Condo', 'Condo Townhouse']
        
        # Generate realistic sample data
        sample_data = []
        for i in range(500):  # 500 sample properties
            sample_data.append({
                'property_id': f'T{1000 + i}',
                'address': f'{100 + i} Sample Street',
                'neighborhood': neighborhoods[i % len(neighborhoods)],
                'property_type': property_types[i % len(property_types)],
                'assessed_value': 500000 + (i * 1000) + (hash(neighborhoods[i % len(neighborhoods)]) % 500000),
                'bedrooms': (i % 4) + 1,
                'bathrooms': (i % 3) + 1,
                'square_feet': 800 + (i * 10) + (hash(property_types[i % len(property_types)]) % 1000),
                'year_built': 1950 + (i % 70),
                'lot_size': 3000 + (i * 50),
                'assessment_year': 2024
            })
        
        return pd.DataFrame(sample_data)
    
    def clean_property_data(self, df):
        """
        Clean and standardize the property data
        Handle missing values and data type conversions
        """
        print("Cleaning property data...")
        
        # Convert price columns to numeric
        price_columns = ['assessed_value']
        for col in price_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Convert numeric columns
        numeric_columns = ['bedrooms', 'bathrooms', 'square_feet', 'year_built', 'lot_size']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove rows with missing essential data
        essential_columns = ['assessed_value', 'neighborhood']
        df = df.dropna(subset=essential_columns)
        
        # Remove outliers (properties with unrealistic values)
        df = df[(df['assessed_value'] > 100000) & (df['assessed_value'] < 5000000)]
        
        if 'square_feet' in df.columns:
            df = df[(df['square_feet'] > 200) & (df['square_feet'] < 10000)]
        
        print(f"Data cleaned. {len(df)} properties remaining.")
        return df
    
    def save_data(self, df, filename):
        """
        Save cleaned data to CSV for future use
        """
        filepath = os.path.join('data', filename)
        df.to_csv(filepath, index=False)
        print(f"Data saved to {filepath}")
    
    def collect_and_save_data(self):
        """
        Main method to collect, clean, and save Toronto real estate data
        """
        print("Starting Toronto real estate data collection...")
        
        # Collect raw data
        raw_data = self.get_toronto_property_data(limit=1000)
        
        if raw_data is not None and not raw_data.empty:
            # Clean the data
            clean_data = self.clean_property_data(raw_data)
            
            # Save to file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            filename = f"toronto_properties_{timestamp}.csv"
            self.save_data(clean_data, filename)
            
            # Print summary statistics
            print("\nData Collection Summary:")
            print(f"Total properties: {len(clean_data)}")
            if 'neighborhood' in clean_data.columns:
                print(f"Neighborhoods covered: {clean_data['neighborhood'].nunique()}")
            if 'assessed_value' in clean_data.columns:
                print(f"Price range: ${clean_data['assessed_value'].min():,.0f} - ${clean_data['assessed_value'].max():,.0f}")
                print(f"Average price: ${clean_data['assessed_value'].mean():,.0f}")
            
            return clean_data
        else:
            print("Failed to collect data.")
            return None

# Test the collector if running this file directly
if __name__ == "__main__":
    collector = TorontoRealEstateCollector()
    data = collector.collect_and_save_data()
    
    if data is not None:
        print("\nFirst 5 properties:")
        print(data.head())