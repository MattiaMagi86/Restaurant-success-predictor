#!/usr/bin/env python3
"""
Restaurant Success Predictor - Data Science Portfolio Project
Created by: Mattia Magi

This project demonstrates:
1. Data Analysis capabilities
2. Machine Learning for predictions
3. Business insights from hospitality industry knowledge
4. Data visualization and reporting

Business Problem: 
Can we predict restaurant success based on location, menu pricing, and operational factors?
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

def generate_restaurant_data(n_samples=1000):
    """
    Generate synthetic restaurant data based on real-world hospitality insights
    This simulates the type of data analysis work done at Prosocial Media
    """
    print("🍽️  Generating Restaurant Performance Dataset...")
    
    # Location types (based on real hospitality experience)
    locations = ['Downtown', 'Suburban', 'Tourist Area', 'Business District', 'Residential']
    location_weights = [0.25, 0.30, 0.15, 0.20, 0.10]
    
    # Restaurant types
    cuisines = ['Italian', 'Mexican', 'Asian', 'American', 'Mediterranean', 'Fast Food']
    cuisine_weights = [0.20, 0.15, 0.20, 0.25, 0.10, 0.10]
    
    data = []
    
    for i in range(n_samples):
        # Basic restaurant characteristics
        location = np.random.choice(locations, p=location_weights)
        cuisine = np.random.choice(cuisines, p=cuisine_weights)
        
        # Generate features based on hospitality industry knowledge
        # Average menu price (influenced by location and cuisine)
        base_price = np.random.normal(18, 5)
        if location == 'Downtown': base_price *= 1.4
        elif location == 'Tourist Area': base_price *= 1.3
        elif location == 'Business District': base_price *= 1.2
        if cuisine == 'Fast Food': base_price *= 0.4
        elif cuisine == 'Mediterranean': base_price *= 1.3
        
        avg_menu_price = max(8, min(50, base_price))
        
        # Operational factors
        staff_count = np.random.randint(5, 25)
        seating_capacity = np.random.randint(30, 150)
        years_open = np.random.randint(1, 15)
        
        # Marketing factors
        social_media_followers = np.random.exponential(1000)
        online_reviews_count = np.random.poisson(50)
        avg_rating = np.random.normal(4.0, 0.8)
        avg_rating = max(1.0, min(5.0, avg_rating))
        
        # Financial factors
        monthly_rent = np.random.normal(8000, 2000)
        if location == 'Downtown': monthly_rent *= 1.8
        elif location == 'Tourist Area': monthly_rent *= 1.5
        monthly_rent = max(3000, monthly_rent)
        
        # Calculate success probability based on industry factors
        success_prob = 0.3  # Base probability
        
        # Price positioning impact
        if 12 <= avg_menu_price <= 25: success_prob += 0.2
        elif avg_menu_price > 35: success_prob -= 0.1
        
        # Location impact
        if location in ['Downtown', 'Tourist Area']: success_prob += 0.15
        elif location == 'Residential': success_prob -= 0.1
        
        # Rating impact (critical in hospitality)
        if avg_rating >= 4.5: success_prob += 0.3
        elif avg_rating >= 4.0: success_prob += 0.15
        elif avg_rating < 3.5: success_prob -= 0.2
        
        # Reviews count impact
        if online_reviews_count > 100: success_prob += 0.1
        elif online_reviews_count < 20: success_prob -= 0.1
        
        # Social media impact
        if social_media_followers > 2000: success_prob += 0.1
        
        # Rent efficiency impact
        rent_per_seat = monthly_rent / seating_capacity
        if rent_per_seat < 60: success_prob += 0.1
        elif rent_per_seat > 100: success_prob -= 0.15
        
        # Years open impact (experience matters)
        if years_open >= 5: success_prob += 0.1
        elif years_open >= 10: success_prob += 0.05
        
        # Cap probability
        success_prob = max(0.05, min(0.95, success_prob))
        
        # Determine success
        is_successful = np.random.random() < success_prob
        
        data.append({
            'location': location,
            'cuisine_type': cuisine,
            'avg_menu_price': round(avg_menu_price, 2),
            'staff_count': staff_count,
            'seating_capacity': seating_capacity,
            'years_open': years_open,
            'social_media_followers': int(social_media_followers),
            'online_reviews_count': online_reviews_count,
            'avg_rating': round(avg_rating, 1),
            'monthly_rent': round(monthly_rent, 0),
            'rent_per_seat': round(monthly_rent / seating_capacity, 2),
            'is_successful': is_successful
        })
    
    df = pd.DataFrame(data)
    print(f"✅ Generated {len(df)} restaurant records")
    return df

def perform_exploratory_analysis(df):
    """
    Comprehensive data analysis - demonstrating analytical skills
    """
    print("\n📊 EXPLORATORY DATA ANALYSIS")
    print("=" * 50)
    
    # Basic statistics
    print("\n🔍 Dataset Overview:")
    print(f"Total restaurants: {len(df)}")
    print(f"Successful restaurants: {df['is_successful'].sum()} ({df['is_successful'].mean():.1%})")
    print(f"Failed restaurants: {(~df['is_successful']).sum()} ({(~df['is_successful']).mean():.1%})")
    
    # Success rate by location
    print("\n🏢 Success Rate by Location:")
    location_analysis = df.groupby('location')['is_successful'].agg(['count', 'sum', 'mean']).round(3)
    location_analysis.columns = ['Total', 'Successful', 'Success_Rate']
    print(location_analysis.sort_values('Success_Rate', ascending=False))
    
    # Success rate by cuisine
    print("\n🍝 Success Rate by Cuisine Type:")
    cuisine_analysis = df.groupby('cuisine_type')['is_successful'].agg(['count', 'sum', 'mean']).round(3)
    cuisine_analysis.columns = ['Total', 'Successful', 'Success_Rate']
    print(cuisine_analysis.sort_values('Success_Rate', ascending=False))
    
    # Price analysis
    print("\n💰 Price Analysis:")
    successful_prices = df[df['is_successful']]['avg_menu_price']
    failed_prices = df[~df['is_successful']]['avg_menu_price']
    print(f"Successful restaurants - Avg price: ${successful_prices.mean():.2f}")
    print(f"Failed restaurants - Avg price: ${failed_prices.mean():.2f}")
    
    # Rating analysis
    print("\n⭐ Rating Analysis:")
    successful_ratings = df[df['is_successful']]['avg_rating']
    failed_ratings = df[~df['is_successful']]['avg_rating']
    print(f"Successful restaurants - Avg rating: {successful_ratings.mean():.2f}")
    print(f"Failed restaurants - Avg rating: {failed_ratings.mean():.2f}")
    
    return df

def create_visualizations(df):
    """
    Create business intelligence visualizations
    """
    print("\n📈 CREATING VISUALIZATIONS")
    print("=" * 50)
    
    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Restaurant Success Analysis Dashboard', fontsize=16, fontweight='bold')
    
    # 1. Success rate by location
    location_success = df.groupby('location')['is_successful'].mean().sort_values(ascending=True)
    axes[0,0].barh(location_success.index, location_success.values, color='skyblue', edgecolor='navy')
    axes[0,0].set_title('Success Rate by Location', fontweight='bold')
    axes[0,0].set_xlabel('Success Rate')
    for i, v in enumerate(location_success.values):
        axes[0,0].text(v + 0.01, i, f'{v:.1%}', va='center')
    
    # 2. Price vs Success
    successful = df[df['is_successful']]
    failed = df[~df['is_successful']]
    axes[0,1].hist([failed['avg_menu_price'], successful['avg_menu_price']], 
                   bins=20, label=['Failed', 'Successful'], alpha=0.7, color=['red', 'green'])
    axes[0,1].set_title('Menu Price Distribution by Success', fontweight='bold')
    axes[0,1].set_xlabel('Average Menu Price ($)')
    axes[0,1].set_ylabel('Count')
    axes[0,1].legend()
    
    # 3. Rating vs Success
    axes[1,0].boxplot([failed['avg_rating'], successful['avg_rating']], 
                      labels=['Failed', 'Successful'])
    axes[1,0].set_title('Rating Distribution by Success', fontweight='bold')
    axes[1,0].set_ylabel('Average Rating')
    
    # 4. Success rate by cuisine
    cuisine_success = df.groupby('cuisine_type')['is_successful'].mean().sort_values(ascending=True)
    axes[1,1].barh(cuisine_success.index, cuisine_success.values, color='lightcoral', edgecolor='darkred')
    axes[1,1].set_title('Success Rate by Cuisine Type', fontweight='bold')
    axes[1,1].set_xlabel('Success Rate')
    for i, v in enumerate(cuisine_success.values):
        axes[1,1].text(v + 0.01, i, f'{v:.1%}', va='center')
    
    plt.tight_layout()
    plt.savefig('restaurant_analysis_dashboard.png', dpi=300, bbox_inches='tight')
    print("✅ Saved visualization dashboard as 'restaurant_analysis_dashboard.png'")
    
    return fig

def build_prediction_models(df):
    """
    Build and compare machine learning models for predicting restaurant success
    """
    print("\n🤖 BUILDING PREDICTION MODELS")
    print("=" * 50)
    
    # Prepare the data
    # Encode categorical variables
    le_location = LabelEncoder()
    le_cuisine = LabelEncoder()
    
    df_model = df.copy()
    df_model['location_encoded'] = le_location.fit_transform(df_model['location'])
    df_model['cuisine_encoded'] = le_cuisine.fit_transform(df_model['cuisine_type'])
    
    # Select features for modeling
    feature_columns = [
        'location_encoded', 'cuisine_encoded', 'avg_menu_price', 
        'staff_count', 'seating_capacity', 'years_open',
        'social_media_followers', 'online_reviews_count', 'avg_rating',
        'monthly_rent', 'rent_per_seat'
    ]
    
    X = df_model[feature_columns]
    y = df_model['is_successful']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale features for logistic regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    # Model 1: Random Forest
    print("\n🌳 Training Random Forest Classifier...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    rf_model.fit(X_train, y_train)
    rf_predictions = rf_model.predict(X_test)
    rf_accuracy = accuracy_score(y_test, rf_predictions)
    
    print(f"Random Forest Accuracy: {rf_accuracy:.3f}")
    
    # Model 2: Logistic Regression
    print("\n📈 Training Logistic Regression...")
    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    lr_model.fit(X_train_scaled, y_train)
    lr_predictions = lr_model.predict(X_test_scaled)
    lr_accuracy = accuracy_score(y_test, lr_predictions)
    
    print(f"Logistic Regression Accuracy: {lr_accuracy:.3f}")
    
    # Feature importance analysis (Random Forest)
    print("\n🔍 FEATURE IMPORTANCE ANALYSIS")
    print("-" * 30)
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("Top factors for restaurant success:")
    for idx, row in feature_importance.head(5).iterrows():
        feature_name = row['feature']
        if feature_name == 'location_encoded':
            feature_name = 'Location'
        elif feature_name == 'cuisine_encoded':
            feature_name = 'Cuisine Type'
        elif feature_name == 'avg_menu_price':
            feature_name = 'Menu Price'
        elif feature_name == 'avg_rating':
            feature_name = 'Customer Rating'
        elif feature_name == 'online_reviews_count':
            feature_name = 'Number of Reviews'
        
        print(f"{feature_name}: {row['importance']:.3f}")
    
    # Model evaluation
    print("\n📋 DETAILED MODEL EVALUATION")
    print("-" * 30)
    print("Random Forest Classification Report:")
    print(classification_report(y_test, rf_predictions))
    
    # Business insights
    print("\n💡 BUSINESS INSIGHTS & RECOMMENDATIONS")
    print("=" * 50)
    
    # Key success factors
    high_rating_success = df[df['avg_rating'] >= 4.5]['is_successful'].mean()
    low_rating_success = df[df['avg_rating'] < 3.5]['is_successful'].mean()
    
    downtown_success = df[df['location'] == 'Downtown']['is_successful'].mean()
    suburban_success = df[df['location'] == 'Suburban']['is_successful'].mean()
    
    print(f"1. CUSTOMER RATINGS MATTER:")
    print(f"   • Restaurants with 4.5+ ratings: {high_rating_success:.1%} success rate")
    print(f"   • Restaurants with <3.5 ratings: {low_rating_success:.1%} success rate")
    print(f"   • Rating improvement can increase success by {(high_rating_success - low_rating_success):.1%}")
    
    print(f"\n2. LOCATION STRATEGY:")
    print(f"   • Downtown locations: {downtown_success:.1%} success rate")
    print(f"   • Suburban locations: {suburban_success:.1%} success rate")
    
    optimal_price_range = df[(df['avg_menu_price'] >= 12) & (df['avg_menu_price'] <= 25)]
    optimal_price_success = optimal_price_range['is_successful'].mean()
    print(f"\n3. PRICING STRATEGY:")
    print(f"   • Optimal price range ($12-$25): {optimal_price_success:.1%} success rate")
    
    return rf_model, lr_model, feature_importance, y_test, rf_predictions, lr_predictions

def create_model_performance_visualizations(feature_importance, y_test, rf_predictions, lr_predictions, df):
    """
    Create additional visualizations for model performance and insights
    """
    print("\n📊 CREATING MODEL PERFORMANCE VISUALIZATIONS")
    print("=" * 50)

    # Create a figure with multiple subplots
    fig = plt.figure(figsize=(16, 10))

    # 1. Feature Importance Chart
    ax1 = plt.subplot(2, 3, 1)
    top_features = feature_importance.head(8).copy()
    # Rename features for better readability
    feature_names_map = {
        'avg_rating': 'Customer Rating',
        'rent_per_seat': 'Rent per Seat',
        'avg_menu_price': 'Menu Price',
        'monthly_rent': 'Monthly Rent',
        'social_media_followers': 'Social Media',
        'online_reviews_count': 'Online Reviews',
        'location_encoded': 'Location',
        'cuisine_encoded': 'Cuisine Type',
        'seating_capacity': 'Seating Capacity',
        'staff_count': 'Staff Count',
        'years_open': 'Years Open'
    }
    top_features['feature'] = top_features['feature'].map(lambda x: feature_names_map.get(x, x))

    colors = plt.cm.viridis(top_features['importance'] / top_features['importance'].max())
    bars = ax1.barh(top_features['feature'], top_features['importance'], color=colors)
    ax1.set_xlabel('Importance Score', fontweight='bold')
    ax1.set_title('Top Success Factors', fontweight='bold', fontsize=12)
    ax1.invert_yaxis()
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax1.text(width, bar.get_y() + bar.get_height()/2, f'{width:.3f}',
                ha='left', va='center', fontsize=9, fontweight='bold')

    # 2. Confusion Matrix for Random Forest
    ax2 = plt.subplot(2, 3, 2)
    cm = confusion_matrix(y_test, rf_predictions)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2, cbar=False)
    ax2.set_title('Random Forest - Confusion Matrix', fontweight='bold', fontsize=12)
    ax2.set_ylabel('Actual', fontweight='bold')
    ax2.set_xlabel('Predicted', fontweight='bold')
    ax2.set_xticklabels(['Failed', 'Successful'])
    ax2.set_yticklabels(['Failed', 'Successful'])

    # 3. Confusion Matrix for Logistic Regression
    ax3 = plt.subplot(2, 3, 3)
    cm_lr = confusion_matrix(y_test, lr_predictions)
    sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Greens', ax=ax3, cbar=False)
    ax3.set_title('Logistic Regression - Confusion Matrix', fontweight='bold', fontsize=12)
    ax3.set_ylabel('Actual', fontweight='bold')
    ax3.set_xlabel('Predicted', fontweight='bold')
    ax3.set_xticklabels(['Failed', 'Successful'])
    ax3.set_yticklabels(['Failed', 'Successful'])

    # 4. Success Rate by Rating Buckets
    ax4 = plt.subplot(2, 3, 4)
    df_plot = df.copy()
    df_plot['rating_bucket'] = pd.cut(df_plot['avg_rating'], bins=[0, 3, 3.5, 4, 4.5, 5],
                                       labels=['<3.0', '3.0-3.5', '3.5-4.0', '4.0-4.5', '4.5+'])
    rating_success = df_plot.groupby('rating_bucket')['is_successful'].mean()
    bars = ax4.bar(rating_success.index, rating_success.values, color='coral', edgecolor='darkred', linewidth=2)
    ax4.set_title('Success Rate by Rating Range', fontweight='bold', fontsize=12)
    ax4.set_xlabel('Rating Range', fontweight='bold')
    ax4.set_ylabel('Success Rate', fontweight='bold')
    ax4.set_ylim(0, 1)
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1%}', ha='center', va='bottom', fontweight='bold')

    # 5. Price vs Rating Scatter (Success/Failure)
    ax5 = plt.subplot(2, 3, 5)
    successful = df[df['is_successful']]
    failed = df[~df['is_successful']]
    ax5.scatter(failed['avg_menu_price'], failed['avg_rating'],
               alpha=0.5, c='red', label='Failed', s=30)
    ax5.scatter(successful['avg_menu_price'], successful['avg_rating'],
               alpha=0.5, c='green', label='Successful', s=30)
    ax5.set_title('Price vs Rating by Success Status', fontweight='bold', fontsize=12)
    ax5.set_xlabel('Average Menu Price ($)', fontweight='bold')
    ax5.set_ylabel('Average Rating', fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # 6. Success Rate by Years Open
    ax6 = plt.subplot(2, 3, 6)
    df_plot['years_bucket'] = pd.cut(df_plot['years_open'], bins=[0, 2, 5, 10, 20],
                                      labels=['0-2 yrs', '2-5 yrs', '5-10 yrs', '10+ yrs'])
    years_success = df_plot.groupby('years_bucket')['is_successful'].mean()
    bars = ax6.bar(years_success.index, years_success.values, color='lightblue', edgecolor='navy', linewidth=2)
    ax6.set_title('Success Rate by Years in Business', fontweight='bold', fontsize=12)
    ax6.set_xlabel('Years Open', fontweight='bold')
    ax6.set_ylabel('Success Rate', fontweight='bold')
    ax6.set_ylim(0, 1)
    for bar in bars:
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1%}', ha='center', va='bottom', fontweight='bold')

    plt.suptitle('Restaurant Success Prediction - Model Performance & Insights',
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('model_performance_analysis.png', dpi=300, bbox_inches='tight')
    print("✅ Saved model performance visualizations as 'model_performance_analysis.png'")

    return fig

def generate_business_report(df, feature_importance):
    """
    Generate a comprehensive business report
    """
    print("\n📊 EXECUTIVE SUMMARY REPORT")
    print("=" * 50)
    
    total_restaurants = len(df)
    successful_restaurants = df['is_successful'].sum()
    success_rate = df['is_successful'].mean()
    
    report = f"""
    RESTAURANT SUCCESS ANALYSIS - EXECUTIVE SUMMARY
    
    📈 OVERALL PERFORMANCE METRICS:
    • Total restaurants analyzed: {total_restaurants:,}
    • Successful restaurants: {successful_restaurants:,}
    • Overall success rate: {success_rate:.1%}
    
    🎯 KEY SUCCESS FACTORS (by importance):
    """
    
    for idx, row in feature_importance.head(3).iterrows():
        feature_name = row['feature']
        if feature_name == 'avg_rating':
            report += f"\n    1. Customer Rating (Impact: {row['importance']:.1%})"
        elif feature_name == 'location_encoded':
            report += f"\n    2. Location Choice (Impact: {row['importance']:.1%})"
        elif feature_name == 'avg_menu_price':
            report += f"\n    3. Menu Pricing (Impact: {row['importance']:.1%})"
    
    # Best performing segments
    best_location = df.groupby('location')['is_successful'].mean().idxmax()
    best_location_rate = df.groupby('location')['is_successful'].mean().max()
    
    best_cuisine = df.groupby('cuisine_type')['is_successful'].mean().idxmax()
    best_cuisine_rate = df.groupby('cuisine_type')['is_successful'].mean().max()
    
    report += f"""
    
    🏆 TOP PERFORMING SEGMENTS:
    • Best location: {best_location} ({best_location_rate:.1%} success rate)
    • Best cuisine type: {best_cuisine} ({best_cuisine_rate:.1%} success rate)
    
    💰 FINANCIAL INSIGHTS:
    • Average successful restaurant menu price: ${df[df['is_successful']]['avg_menu_price'].mean():.2f}
    • Average failed restaurant menu price: ${df[~df['is_successful']]['avg_menu_price'].mean():.2f}
    
    ⭐ CUSTOMER EXPERIENCE:
    • Average successful restaurant rating: {df[df['is_successful']]['avg_rating'].mean():.1f}/5.0
    • Average failed restaurant rating: {df[~df['is_successful']]['avg_rating'].mean():.1f}/5.0
    
    🎯 ACTIONABLE RECOMMENDATIONS:
    1. Focus on customer experience - ratings are the #1 success predictor
    2. Choose location strategically - {best_location} shows highest success rates
    3. Price competitively in the $12-25 range for optimal market positioning
    4. Invest in online presence - reviews and social media drive success
    5. Monitor rent-to-capacity ratio - keep under $60 per seat monthly
    """
    
    print(report)
    
    # Save report to file
    with open('restaurant_success_report.txt', 'w') as f:
        f.write(report)
    
    print("\n✅ Full report saved as 'restaurant_success_report.txt'")
    
    return report

def main():
    """
    Main execution function - demonstrates complete data science workflow
    """
    print("🚀 RESTAURANT SUCCESS PREDICTOR - DATA SCIENCE PROJECT")
    print("Created by: Mattia Magi")
    print("Demonstrates: Data Analysis • Machine Learning • Business Intelligence")
    print("=" * 70)
    
    # Step 1: Data Generation
    df = generate_restaurant_data(1000)
    
    # Step 2: Exploratory Data Analysis
    df = perform_exploratory_analysis(df)
    
    # Step 3: Data Visualization
    create_visualizations(df)
    
    # Step 4: Machine Learning Models
    rf_model, lr_model, feature_importance, y_test, rf_predictions, lr_predictions = build_prediction_models(df)

    # Step 4.5: Create Model Performance Visualizations
    create_model_performance_visualizations(feature_importance, y_test, rf_predictions, lr_predictions, df)

    # Step 5: Business Report
    generate_business_report(df, feature_importance)
    
    # Step 6: Save processed data
    df.to_csv('restaurant_data.csv', index=False)
    print(f"\n✅ Dataset saved as 'restaurant_data.csv'")
    
    print("\n🎉 PROJECT COMPLETED SUCCESSFULLY!")
    print("\nThis project demonstrates:")
    print("✓ Data generation and preprocessing")
    print("✓ Exploratory data analysis")
    print("✓ Statistical analysis and insights")
    print("✓ Machine learning model building")
    print("✓ Feature importance analysis")
    print("✓ Business intelligence and reporting")
    print("✓ Data visualization and dashboards")
    print("✓ Industry domain knowledge application")
    
    return df, rf_model, lr_model

if __name__ == "__main__":
    # Run the complete analysis
    restaurant_data, random_forest_model, logistic_model = main()
