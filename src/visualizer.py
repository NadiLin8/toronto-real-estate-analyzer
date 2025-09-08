import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import glob
from datetime import datetime

class RealEstateVisualizer:
    def __init__(self, data_file=None):
        """
        Initialize visualizer with property data
        Sets up plotting style and loads data
        """
        # Set up professional plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create output directory for charts
        if not os.path.exists('charts'):
            os.makedirs('charts')
        
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
        
        print(f"Creating visualizations for {len(self.data)} properties")
    
    def create_price_distribution_chart(self):
        """
        Create histogram showing property price distribution
        Shows market price ranges and concentration
        """
        plt.figure(figsize=(12, 6))
        
        # Create histogram with density curve
        plt.hist(self.data['assessed_value'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        
        # Add mean and median lines
        mean_price = self.data['assessed_value'].mean()
        median_price = self.data['assessed_value'].median()
        
        plt.axvline(mean_price, color='red', linestyle='--', linewidth=2, label=f'Mean: ${mean_price:,.0f}')
        plt.axvline(median_price, color='green', linestyle='--', linewidth=2, label=f'Median: ${median_price:,.0f}')
        
        plt.title('Toronto Real Estate Price Distribution', fontsize=16, fontweight='bold')
        plt.xlabel('Property Value ($)', fontsize=12)
        plt.ylabel('Number of Properties', fontsize=12)
        plt.legend()
        
        # Format x-axis to show prices in readable format
        plt.ticklabel_format(style='plain', axis='x')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig('charts/price_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("✓ Price distribution chart saved to charts/price_distribution.png")
    
    def create_neighborhood_comparison_chart(self):
        """
        Create bar chart comparing average prices by neighborhood
        Helps identify premium vs affordable areas
        """
        # Calculate neighborhood statistics
        neighborhood_stats = self.data.groupby('neighborhood')['assessed_value'].agg(['mean', 'count']).round(0)
        neighborhood_stats = neighborhood_stats.sort_values('mean', ascending=True)
        
        plt.figure(figsize=(14, 8))
        
        # Create horizontal bar chart for better readability
        bars = plt.barh(neighborhood_stats.index, neighborhood_stats['mean'], color='lightcoral')
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            plt.text(width + 10000, bar.get_y() + bar.get_height()/2, 
                    f'${width:,.0f}', ha='left', va='center', fontweight='bold')
        
        plt.title('Average Property Values by Toronto Neighborhood', fontsize=16, fontweight='bold')
        plt.xlabel('Average Property Value ($)', fontsize=12)
        plt.ylabel('Neighborhood', fontsize=12)
        
        # Format x-axis
        plt.ticklabel_format(style='plain', axis='x')
        
        plt.tight_layout()
        plt.savefig('charts/neighborhood_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("✓ Neighborhood comparison chart saved to charts/neighborhood_comparison.png")
    
    def create_property_type_analysis_chart(self):
        """
        Create charts comparing different property types
        Shows value differences between condos, houses, etc.
        """
        if 'property_type' not in self.data.columns:
            print("Property type data not available for visualization.")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Chart 1: Average price by property type
        type_stats = self.data.groupby('property_type')['assessed_value'].mean().sort_values(ascending=False)
        
        bars1 = ax1.bar(type_stats.index, type_stats.values, color='lightgreen')
        ax1.set_title('Average Price by Property Type', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Average Price ($)', fontsize=12)
        ax1.set_xlabel('Property Type', fontsize=12)
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 5000,
                    f'${height:,.0f}', ha='center', va='bottom', fontweight='bold')
        
        # Chart 2: Property count by type
        type_counts = self.data['property_type'].value_counts()
        
        ax2.pie(type_counts.values, labels=type_counts.index, autopct='%1.1f%%', startangle=90)
        ax2.set_title('Distribution of Property Types', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('charts/property_type_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("✓ Property type analysis chart saved to charts/property_type_analysis.png")
    
    def create_price_per_sqft_chart(self):
        """
        Create scatter plot showing price per square foot analysis
        Helps identify value opportunities
        """
        if 'square_feet' not in self.data.columns:
            print("Square footage data not available for price per sqft analysis.")
            return
        
        # Calculate price per square foot
        self.data['price_per_sqft'] = self.data['assessed_value'] / self.data['square_feet']
        
        plt.figure(figsize=(12, 8))
        
        # Create scatter plot colored by neighborhood
        neighborhoods = self.data['neighborhood'].unique()
        colors = plt.cm.Set3(np.linspace(0, 1, len(neighborhoods)))
        
        for i, neighborhood in enumerate(neighborhoods):
            subset = self.data[self.data['neighborhood'] == neighborhood]
            plt.scatter(subset['square_feet'], subset['price_per_sqft'], 
                       c=[colors[i]], label=neighborhood, alpha=0.7, s=50)
        
        plt.title('Price per Square Foot vs Property Size by Neighborhood', fontsize=16, fontweight='bold')
        plt.xlabel('Square Feet', fontsize=12)
        plt.ylabel('Price per Square Foot ($)', fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Add trend line
        z = np.polyfit(self.data['square_feet'], self.data['price_per_sqft'], 1)
        p = np.poly1d(z)
        plt.plot(self.data['square_feet'], p(self.data['square_feet']), "r--", alpha=0.8, linewidth=2)
        
        plt.tight_layout()
        plt.savefig('charts/price_per_sqft_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("✓ Price per square foot chart saved to charts/price_per_sqft_analysis.png")
    
    def create_market_heatmap(self):
        """
        Create correlation heatmap of property features
        Shows which factors most influence property values
        """
        # Select numeric columns for correlation analysis
        numeric_columns = ['assessed_value', 'bedrooms', 'bathrooms', 'square_feet', 'year_built', 'lot_size']
        available_columns = [col for col in numeric_columns if col in self.data.columns]
        
        if len(available_columns) < 3:
            print("Insufficient numeric data for correlation heatmap.")
            return
        
        # Calculate correlation matrix
        correlation_matrix = self.data[available_columns].corr()
        
        plt.figure(figsize=(10, 8))
        
        # Create heatmap
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))  # Hide upper triangle
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": .8})
        
        plt.title('Property Features Correlation Heatmap', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('charts/correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("✓ Correlation heatmap saved to charts/correlation_heatmap.png")
    
    def create_investment_opportunity_chart(self):
        """
        Create chart highlighting potential investment opportunities
        Combines price and value metrics
        """
        # Calculate key metrics by neighborhood
        neighborhood_metrics = self.data.groupby('neighborhood').agg({
            'assessed_value': ['mean', 'count'],
            'square_feet': 'mean',
            'year_built': 'mean'
        }).round(0)
        
        # Flatten column names
        neighborhood_metrics.columns = ['avg_price', 'property_count', 'avg_sqft', 'avg_year_built']
        
        # Calculate price per sqft
        neighborhood_metrics['price_per_sqft'] = neighborhood_metrics['avg_price'] / neighborhood_metrics['avg_sqft']
        
        # Create opportunity score (lower price per sqft + newer buildings = better opportunity)
        neighborhood_metrics['opportunity_score'] = (
            (1 / neighborhood_metrics['price_per_sqft']) * 1000 +  # Inverse of price per sqft
            (neighborhood_metrics['avg_year_built'] - 1950) / 10   # Newer is better
        )
        
        plt.figure(figsize=(12, 8))
        
        # Create bubble chart
        x = neighborhood_metrics['avg_price']
        y = neighborhood_metrics['price_per_sqft']
        sizes = neighborhood_metrics['opportunity_score'] * 5  # Scale for visibility
        
        scatter = plt.scatter(x, y, s=sizes, alpha=0.6, c=neighborhood_metrics['opportunity_score'], 
                            cmap='RdYlGn', edgecolors='black', linewidth=1)
        
        # Add neighborhood labels
        for i, neighborhood in enumerate(neighborhood_metrics.index):
            plt.annotate(neighborhood, (x.iloc[i], y.iloc[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=10)
        
        plt.title('Investment Opportunity Analysis by Neighborhood', fontsize=16, fontweight='bold')
        plt.xlabel('Average Property Price ($)', fontsize=12)
        plt.ylabel('Price per Square Foot ($)', fontsize=12)
        
        # Add colorbar
        cbar = plt.colorbar(scatter)
        cbar.set_label('Investment Opportunity Score (Higher = Better)', fontsize=10)
        
        plt.ticklabel_format(style='plain', axis='x')
        plt.tight_layout()
        plt.savefig('charts/investment_opportunities.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("✓ Investment opportunity chart saved to charts/investment_opportunities.png")
    
    def generate_all_visualizations(self):
        """
        Create complete set of visualizations for the real estate analysis
        Professional dashboard-style charts
        """
        print("\n" + "="*60)
        print("GENERATING REAL ESTATE VISUALIZATION DASHBOARD")
        print("="*60)
        
        try:
            self.create_price_distribution_chart()
            self.create_neighborhood_comparison_chart()
            self.create_property_type_analysis_chart()
            self.create_price_per_sqft_chart()
            self.create_market_heatmap()
            self.create_investment_opportunity_chart()
            
            print("\n" + "="*60)
            print("VISUALIZATION DASHBOARD COMPLETE!")
            print("="*60)
            print("All charts saved to the 'charts/' directory:")
            print("• price_distribution.png - Market price ranges")
            print("• neighborhood_comparison.png - Area comparisons") 
            print("• property_type_analysis.png - Property type breakdown")
            print("• price_per_sqft_analysis.png - Value analysis")
            print("• correlation_heatmap.png - Feature relationships")
            print("• investment_opportunities.png - Investment insights")
            print("\nThese professional charts are perfect for presentations and portfolios!")
            
        except Exception as e:
            print(f"Error generating visualizations: {str(e)}")

# Test the visualizer if running this file directly
if __name__ == "__main__":
    try:
        visualizer = RealEstateVisualizer()
        visualizer.generate_all_visualizations()
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run 'python data_collector.py' first to collect data.")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")