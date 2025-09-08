import pandas as pd
import numpy as np
from datetime import datetime
import os
import glob

class RealEstateAnalyzer:
    def __init__(self, data_file=None):
        """
        Initialize analyzer with property data
        If no file specified, loads the most recent data file
        """
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
        
        print(f"Analyzing {len(self.data)} properties")
    
    def market_overview(self):
        """
        Generate comprehensive market overview statistics
        Key metrics that investors and analysts care about
        """
        print("\n" + "="*50)
        print("TORONTO REAL ESTATE MARKET OVERVIEW")
        print("="*50)
        
        overview = {
            'total_properties': len(self.data),
            'avg_price': self.data['assessed_value'].mean(),
            'median_price': self.data['assessed_value'].median(),
            'price_std': self.data['assessed_value'].std(),
            'min_price': self.data['assessed_value'].min(),
            'max_price': self.data['assessed_value'].max()
        }
        
        print(f"Total Properties Analyzed: {overview['total_properties']:,}")
        print(f"Average Property Value: ${overview['avg_price']:,.0f}")
        print(f"Median Property Value: ${overview['median_price']:,.0f}")
        print(f"Price Standard Deviation: ${overview['price_std']:,.0f}")
        print(f"Price Range: ${overview['min_price']:,.0f} - ${overview['max_price']:,.0f}")
        
        # Calculate price per square foot if data available
        if 'square_feet' in self.data.columns:
            self.data['price_per_sqft'] = self.data['assessed_value'] / self.data['square_feet']
            avg_price_per_sqft = self.data['price_per_sqft'].mean()
            print(f"Average Price per Sq Ft: ${avg_price_per_sqft:.0f}")
        
        return overview
    
    def neighborhood_analysis(self):
        """
        Analyze market trends by neighborhood
        Identifies hot markets and value opportunities
        """
        print("\n" + "="*50)
        print("NEIGHBORHOOD MARKET ANALYSIS")
        print("="*50)
        
        # Group by neighborhood and calculate key metrics
        neighborhood_stats = self.data.groupby('neighborhood').agg({
            'assessed_value': ['count', 'mean', 'median', 'std'],
            'square_feet': 'mean',
            'year_built': 'mean'
        }).round(0)
        
        # Flatten column names
        neighborhood_stats.columns = ['property_count', 'avg_price', 'median_price', 
                                    'price_std', 'avg_sqft', 'avg_year_built']
        
        # Calculate price per sqft by neighborhood
        if 'square_feet' in self.data.columns:
            neighborhood_stats['price_per_sqft'] = neighborhood_stats['avg_price'] / neighborhood_stats['avg_sqft']
        
        # Sort by average price (descending)
        neighborhood_stats = neighborhood_stats.sort_values('avg_price', ascending=False)
        
        print("Neighborhoods ranked by average property value:")
        print(neighborhood_stats.to_string())
        
        # Identify investment opportunities
        print("\n" + "-"*50)
        print("INVESTMENT OPPORTUNITY ANALYSIS")
        print("-"*50)
        
        # Find neighborhoods with low price but high potential
        overall_median = self.data['assessed_value'].median()
        
        opportunities = []
        for neighborhood in neighborhood_stats.index:
            stats = neighborhood_stats.loc[neighborhood]
            
            # Criteria for good investment: below median price, newer buildings, good size
            if (stats['avg_price'] < overall_median * 1.1 and  # Not too expensive
                stats['avg_year_built'] > 1980 and             # Relatively newer
                stats['property_count'] >= 10):                # Sufficient data
                
                opportunities.append({
                    'neighborhood': neighborhood,
                    'avg_price': stats['avg_price'],
                    'property_count': int(stats['property_count']),
                    'reason': 'Below market average, newer construction'
                })
        
        if opportunities:
            print("\nPotential Investment Opportunities:")
            for opp in opportunities:
                print(f"• {opp['neighborhood']}: Avg ${opp['avg_price']:,.0f} ({opp['property_count']} properties)")
                print(f"  Reason: {opp['reason']}")
        else:
            print("No clear investment opportunities identified in current data.")
        
        return neighborhood_stats
    
    def property_type_analysis(self):
        """
        Compare different property types (Detached, Condo, etc.)
        Helps understand which property types offer best value
        """
        print("\n" + "="*50)
        print("PROPERTY TYPE ANALYSIS")
        print("="*50)
        
        if 'property_type' not in self.data.columns:
            print("Property type data not available.")
            return None
        
        type_stats = self.data.groupby('property_type').agg({
            'assessed_value': ['count', 'mean', 'median'],
            'square_feet': 'mean',
            'bedrooms': 'mean'
        }).round(0)
        
        # Flatten column names
        type_stats.columns = ['count', 'avg_price', 'median_price', 'avg_sqft', 'avg_bedrooms']
        
        # Calculate value metrics
        if 'square_feet' in self.data.columns:
            type_stats['price_per_sqft'] = type_stats['avg_price'] / type_stats['avg_sqft']
        
        # Sort by average price
        type_stats = type_stats.sort_values('avg_price', ascending=False)
        
        print("Property types ranked by average value:")
        print(type_stats.to_string())
        
        # Find best value property type
        if 'price_per_sqft' in type_stats.columns:
            best_value = type_stats['price_per_sqft'].idxmin()
            print(f"\nBest Value Property Type: {best_value}")
            print(f"Price per Sq Ft: ${type_stats.loc[best_value, 'price_per_sqft']:.0f}")
        
        return type_stats
    
    def market_trends_analysis(self):
        """
        Analyze trends based on property age and characteristics
        Identifies patterns in the market
        """
        print("\n" + "="*50)
        print("MARKET TRENDS ANALYSIS")
        print("="*50)
        
        if 'year_built' not in self.data.columns:
            print("Year built data not available for trend analysis.")
            return None
        
        # Create age categories
        current_year = datetime.now().year
        self.data['property_age'] = current_year - self.data['year_built']
        
        # Define age brackets
        age_brackets = [
            (0, 10, 'New (0-10 years)'),
            (11, 25, 'Modern (11-25 years)'),
            (26, 50, 'Mature (26-50 years)'),
            (51, 100, 'Older (51+ years)')
        ]
        
        print("Property values by age:")
        for min_age, max_age, label in age_brackets:
            mask = (self.data['property_age'] >= min_age) & (self.data['property_age'] <= max_age)
            subset = self.data[mask]
            
            if len(subset) > 0:
                avg_price = subset['assessed_value'].mean()
                count = len(subset)
                print(f"{label}: ${avg_price:,.0f} average ({count} properties)")
        
        # Analyze bedroom count vs price
        if 'bedrooms' in self.data.columns:
            print("\nProperty values by bedroom count:")
            bedroom_analysis = self.data.groupby('bedrooms')['assessed_value'].agg(['count', 'mean']).round(0)
            bedroom_analysis.columns = ['count', 'avg_price']
            
            for bedrooms in sorted(bedroom_analysis.index):
                stats = bedroom_analysis.loc[bedrooms]
                print(f"{int(bedrooms)} bedrooms: ${int(stats['avg_price']):,} average ({int(stats['count'])} properties)")
        
        return self.data
    
    def generate_market_report(self):
        """
        Generate comprehensive market analysis report
        Combines all analysis methods into one report
        """
        print("\n" + "#"*60)
        print("COMPREHENSIVE TORONTO REAL ESTATE MARKET REPORT")
        print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("#"*60)
        
        # Run all analyses
        overview = self.market_overview()
        neighborhood_data = self.neighborhood_analysis()
        property_type_data = self.property_type_analysis()
        trend_data = self.market_trends_analysis()
        
        # Generate key insights
        print("\n" + "="*50)
        print("KEY MARKET INSIGHTS")
        print("="*50)
        
        insights = []
        
        # Price insights
        if overview['avg_price'] > 1000000:
            insights.append("• Toronto market shows premium pricing with average over $1M")
        
        # Neighborhood insights
        if neighborhood_data is not None:
            most_expensive = neighborhood_data.index[0]
            least_expensive = neighborhood_data.index[-1]
            insights.append(f"• Most expensive neighborhood: {most_expensive}")
            insights.append(f"• Most affordable neighborhood: {least_expensive}")
        
        # Market stability insight
        price_volatility = overview['price_std'] / overview['avg_price']
        if price_volatility < 0.3:
            insights.append("• Market shows relatively stable pricing (low volatility)")
        else:
            insights.append("• Market shows high price volatility - diverse property values")
        
        for insight in insights:
            print(insight)
        
        print("\n" + "="*50)
        print("ANALYSIS COMPLETE")
        print("="*50)
        
        return {
            'overview': overview,
            'neighborhoods': neighborhood_data,
            'property_types': property_type_data,
            'trends': trend_data
        }

# Test the analyzer if running this file directly
if __name__ == "__main__":
    try:
        analyzer = RealEstateAnalyzer()
        report = analyzer.generate_market_report()
        
        print(f"\nAnalysis complete! Check the detailed report above.")
        print("Next step: Run visualizer.py to create charts and graphs.")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run 'python data_collector.py' first to collect data.")