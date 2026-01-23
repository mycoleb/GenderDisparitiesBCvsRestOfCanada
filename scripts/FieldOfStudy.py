#!/usr/bin/env python3
"""
Analyze gender disparities in postsecondary field of study.
This reveals if women are concentrated in lower-paying fields while men dominate high-paying STEM/engineering.
"""
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Known high-paying vs low-paying field categories
HIGH_PAYING_FIELDS = [
    'engineering',
    'computer',
    'mathematics',
    'physical sciences',
    'architecture',
    'business'
]

LOW_PAYING_FIELDS = [
    'education',
    'nursing',
    'social work',
    'humanities',
    'visual',
    'performing arts'
]

def categorize_field(field_name):
    """Categorize field as high-paying, low-paying, or other"""
    field_lower = str(field_name).lower()
    
    for keyword in HIGH_PAYING_FIELDS:
        if keyword in field_lower:
            return 'High-Paying Fields'
    
    for keyword in LOW_PAYING_FIELDS:
        if keyword in field_lower:
            return 'Low-Paying Fields'
    
    return 'Other Fields'


def analyze_field_of_study(csv_path, output_dir='outputs'):
    """Analyze gender disparities in field of study"""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Load data
    df = pd.read_csv(csv_path, low_memory=False)
    
    print("="*80)
    print("FIELD OF STUDY GENDER DISPARITY ANALYSIS")
    print("="*80)
    
    # Show available columns
    print("\nAvailable columns:")
    print(df.columns.tolist())
    
    # Try to identify key columns
    geo_col = None
    gender_col = None
    field_col = None
    value_col = None
    
    for col in df.columns:
        col_lower = col.lower()
        if 'geo' in col_lower or 'province' in col_lower:
            geo_col = col
        if 'sex' in col_lower or 'gender' in col_lower:
            gender_col = col
        if 'field' in col_lower or 'program' in col_lower:
            field_col = col
        if 'value' in col_lower or 'count' in col_lower or 'number' in col_lower:
            value_col = col
    
    print(f"\nDetected columns:")
    print(f"  Geography: {geo_col}")
    print(f"  Gender: {gender_col}")
    print(f"  Field: {field_col}")
    print(f"  Value: {value_col}")
    
    if not all([geo_col, gender_col, field_col, value_col]):
        print("\nWARNING: Could not detect all required columns. Please check the data.")
        print("\nFirst few rows:")
        print(df.head())
        return
    
    # Clean the data
    df[value_col] = pd.to_numeric(df[value_col], errors='coerce')
    df = df.dropna(subset=[value_col])
    
    # Filter to most recent year if there's a time dimension
    if 'REF_DATE' in df.columns:
        latest_year = df['REF_DATE'].max()
        df = df[df['REF_DATE'] == latest_year]
        print(f"\nFiltered to latest year: {latest_year}")
    
    # Show unique values
    print(f"\nUnique genders: {df[gender_col].unique()}")
    print(f"\nUnique geographies (first 10): {df[geo_col].unique()[:10]}")
    print(f"\nUnique fields (first 20):")
    for field in df[field_col].unique()[:20]:
        print(f"  - {field}")
    
    # Filter to BC and aggregate data
    bc_data = df[df[geo_col].str.contains('British Columbia', case=False, na=False)].copy()
    canada_data = df[df[geo_col].str.contains('Canada', case=False, na=False) & 
                     ~df[geo_col].str.contains('British Columbia', case=False, na=False)].copy()
    
    if bc_data.empty:
        print("\nNo BC data found. Analyzing all of Canada instead.")
        bc_data = df.copy()
    
    # Add field category
    bc_data['field_category'] = bc_data[field_col].apply(categorize_field)
    
    # Calculate gender ratios by field
    print("\n" + "="*80)
    print("GENDER DISTRIBUTION BY FIELD (BC)")
    print("="*80)
    
    field_gender = bc_data.groupby([field_col, gender_col])[value_col].sum().unstack(fill_value=0)
    
    # Calculate % female
    if 'Women' in field_gender.columns or 'Females' in field_gender.columns:
        female_col = 'Women' if 'Women' in field_gender.columns else 'Females'
        male_col = 'Men' if 'Men' in field_gender.columns else 'Males'
        
        if female_col in field_gender.columns and male_col in field_gender.columns:
            field_gender['Total'] = field_gender[female_col] + field_gender[male_col]
            field_gender['% Female'] = (field_gender[female_col] / field_gender['Total'] * 100).round(1)
            field_gender['% Male'] = (field_gender[male_col] / field_gender['Total'] * 100).round(1)
            
            # Sort by % Female
            field_gender_sorted = field_gender.sort_values('% Female', ascending=False)
            
            print("\nMost Female-Dominated Fields:")
            print(field_gender_sorted[[female_col, male_col, '% Female']].head(10))
            
            print("\nMost Male-Dominated Fields:")
            print(field_gender_sorted[[female_col, male_col, '% Female']].tail(10))
    
    # Analyze by field category
    print("\n" + "="*80)
    print("GENDER DISTRIBUTION BY FIELD CATEGORY (BC)")
    print("="*80)
    
    category_gender = bc_data.groupby(['field_category', gender_col])[value_col].sum().unstack(fill_value=0)
    
    if female_col in category_gender.columns and male_col in category_gender.columns:
        category_gender['Total'] = category_gender[female_col] + category_gender[male_col]
        category_gender['% Female'] = (category_gender[female_col] / category_gender['Total'] * 100).round(1)
        category_gender['% Male'] = (category_gender[male_col] / category_gender['Total'] * 100).round(1)
        
        print("\n", category_gender)
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Top fields by gender
    ax1 = axes[0, 0]
    top_fields = field_gender.nlargest(15, 'Total')[[female_col, male_col]]
    top_fields.plot(kind='barh', ax=ax1, color=['#FF69B4', '#4169E1'])
    ax1.set_title('Top 15 Fields by Total Graduates - Gender Breakdown (BC)', fontweight='bold', fontsize=12)
    ax1.set_xlabel('Number of Graduates')
    ax1.legend(title='Gender')
    ax1.grid(axis='x', alpha=0.3)
    
    # 2. % Female by field category
    ax2 = axes[0, 1]
    if '% Female' in category_gender.columns:
        category_gender['% Female'].plot(kind='bar', ax=ax2, color='teal')
        ax2.axhline(y=50, color='red', linestyle='--', label='50% (parity)')
        ax2.set_title('% Female by Field Category (BC)', fontweight='bold', fontsize=12)
        ax2.set_ylabel('% Female')
        ax2.set_xlabel('Field Category')
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
    
    # 3. Most gender-imbalanced fields
    ax3 = axes[1, 0]
    if '% Female' in field_gender.columns:
        # Fields furthest from 50%
        field_gender['Imbalance'] = abs(field_gender['% Female'] - 50)
        most_imbalanced = field_gender.nlargest(10, 'Imbalance')['% Female']
        most_imbalanced.plot(kind='barh', ax=ax3, color='coral')
        ax3.axvline(x=50, color='black', linestyle='--', linewidth=1)
        ax3.set_title('Most Gender-Imbalanced Fields (BC)', fontweight='bold', fontsize=12)
        ax3.set_xlabel('% Female')
        ax3.grid(axis='x', alpha=0.3)
    
    # 4. Compare BC vs Canada (if available)
    ax4 = axes[1, 1]
    if not canada_data.empty:
        canada_data['field_category'] = canada_data[field_col].apply(categorize_field)
        canada_category = canada_data.groupby(['field_category', gender_col])[value_col].sum().unstack(fill_value=0)
        
        if female_col in canada_category.columns and male_col in canada_category.columns:
            canada_category['% Female'] = (canada_category[female_col] / 
                                          (canada_category[female_col] + canada_category[male_col]) * 100)
            
            comparison = pd.DataFrame({
                'BC': category_gender['% Female'],
                'Rest of Canada': canada_category['% Female']
            })
            
            comparison.plot(kind='bar', ax=ax4)
            ax4.axhline(y=50, color='red', linestyle='--', label='50% (parity)')
            ax4.set_title('% Female by Field Category: BC vs Rest of Canada', fontweight='bold', fontsize=12)
            ax4.set_ylabel('% Female')
            ax4.set_xlabel('Field Category')
            ax4.legend()
            ax4.grid(axis='y', alpha=0.3)
            ax4.set_xticklabels(ax4.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'field_of_study_gender_analysis.png', dpi=200, bbox_inches='tight')
    print(f"\n\nVisualization saved to: {output_dir / 'field_of_study_gender_analysis.png'}")
    plt.close()
    
    # Summary statistics
    print("\n" + "="*80)
    print("KEY FINDINGS")
    print("="*80)
    # comparison to determine if women are over represented or underrepresented
    if '% Female' in category_gender.columns:
        high_pay_female = category_gender.loc['High-Paying Fields', '% Female']
        low_pay_female = category_gender.loc['Low-Paying Fields', '% Female']
        
        print(f"\nWomen in High-Paying Fields: {high_pay_female:.1f}%")
        print(f"Women in Low-Paying Fields: {low_pay_female:.1f}%")
        print(f"Disparity: {abs(high_pay_female - low_pay_female):.1f} percentage points")
        
        if high_pay_female < 40:
            print("\n⚠️  SIGNIFICANT DISPARITY: Women are significantly underrepresented in high-paying fields")
        
        if low_pay_female > 60:
            print("⚠️  SIGNIFICANT DISPARITY: Women are overrepresented in low-paying fields")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description='Analyze gender disparities in field of study')
    ap.add_argument('--csv', required=True, help='Path to StatCan CSV file')
    ap.add_argument('--output_dir', default='outputs', help='Output directory for visualizations')
    args = ap.parse_args()
    
    analyze_field_of_study(args.csv, args.output_dir)