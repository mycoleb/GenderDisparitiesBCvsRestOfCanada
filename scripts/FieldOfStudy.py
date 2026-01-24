"""
Analyze gender disparities in postsecondary field of study
Works with StatCan Table 37100168 - handles both aggregate and detailed fields
"""
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

def analyze_field_of_study(csv_path, output_dir='outputs'):
    """Analyze gender disparities in field of study"""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Load data
    print("Loading data...")
    df = pd.read_csv(csv_path, low_memory=False)
    
    print("="*80)
    print("FIELD OF STUDY GENDER DISPARITY ANALYSIS")
    print("="*80)
    
    # Column mapping
    geo_col = 'GEO'
    gender_col = 'Population characteristics'
    field_col = 'Classification of Instructional Program (CIP) 2016'
    value_col = 'VALUE'
    ref_col = 'REF_DATE'
    
    print(f"\nDataset info:")
    print(f"  Total rows: {len(df):,}")
    
    # Clean the data
    df[value_col] = pd.to_numeric(df[value_col], errors='coerce')
    df = df.dropna(subset=[value_col])
    
    # Filter to most recent year
    latest_year = df[ref_col].max()
    df = df[df[ref_col] == latest_year]
    print(f"  Latest year: {latest_year}")
    
    # Filter to gender data only
    gender_data = df[df[gender_col].isin(['Male gender [M]', 'Female gender [F]'])].copy()
    print(f"  Gender data rows: {len(gender_data):,}")
    
    # Filter to Canada-level data
    canada_data = gender_data[gender_data[geo_col] == 'Canada'].copy()
    bc_data = gender_data[gender_data[geo_col] == 'British Columbia'].copy()
    
    # Use Canada data if BC not detailed enough
    if len(bc_data) < len(canada_data) * 0.5:
        print("\n  Using Canada data (BC data limited)")
        data_to_use = canada_data
        region_name = "Canada"
    else:
        print("\n  Using British Columbia data")
        data_to_use = bc_data
        region_name = "British Columbia"
    
    # Check what fields we have
    unique_fields = data_to_use[field_col].unique()
    print(f"\n  Unique fields found: {len(unique_fields)}")
    
    # Pivot to get gender breakdown
    field_gender = data_to_use.groupby([field_col, gender_col])[value_col].sum().unstack(fill_value=0)
    
    female_col = 'Female gender [F]'
    male_col = 'Male gender [M]'
    
    field_gender['Total'] = field_gender[female_col] + field_gender[male_col]
    field_gender['% Female'] = (field_gender[female_col] / field_gender['Total'] * 100).round(1)
    field_gender['% Male'] = (field_gender[male_col] / field_gender['Total'] * 100).round(1)
    
    # Filter out tiny totals
    field_gender = field_gender[field_gender['Total'] > 100]
    
    # Sort by % Female
    field_gender_sorted = field_gender.sort_values('% Female', ascending=False)
    
    print("\n" + "="*80)
    print(f"GENDER DISTRIBUTION BY FIELD ({region_name})")
    print("="*80)
    print(field_gender_sorted[['Total', '% Female', '% Male']].to_string())
    
    # Analysis
    print("\n" + "="*80)
    print("KEY FINDINGS")
    print("="*80)
    
    # Find STEM fields
    stem_fields = field_gender[field_gender.index.str.contains('STEM', case=False, na=False)]
    bhase_fields = field_gender[field_gender.index.str.contains('BHASE', case=False, na=False)]
    
    if not stem_fields.empty:
        stem_row = stem_fields.iloc[0]
        print(f"\n📊 STEM FIELDS:")
        print(f"   Women: {stem_row['% Female']:.1f}% ({stem_row[female_col]:,.0f} graduates)")
        print(f"   Men: {stem_row['% Male']:.1f}% ({stem_row[male_col]:,.0f} graduates)")
        print(f"   Total: {stem_row['Total']:,.0f}")
        
        if stem_row['% Female'] < 45:
            print(f"\n   ⚠️  Women are underrepresented in STEM ({stem_row['% Female']:.1f}% vs 50% parity)")
    
    if not bhase_fields.empty:
        bhase_row = bhase_fields.iloc[0]
        print(f"\n📊 BHASE FIELDS (Business, Humanities, Arts, Social Science, Education):")
        print(f"   Women: {bhase_row['% Female']:.1f}% ({bhase_row[female_col]:,.0f} graduates)")
        print(f"   Men: {bhase_row['% Male']:.1f}% ({bhase_row[male_col]:,.0f} graduates)")
        print(f"   Total: {bhase_row['Total']:,.0f}")
    
    # Create visualization
    print("\n" + "="*80)
    print("CREATING VISUALIZATION")
    print("="*80)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Stacked bar chart
    ax1 = axes[0]
    plot_data = field_gender[[female_col, male_col]].head(10)
    plot_data.columns = ['Women', 'Men']
    plot_data.plot(kind='barh', stacked=True, ax=ax1, color=['#FF1493', '#1E90FF'], width=0.7)
    ax1.set_title(f'Gender Distribution by Field ({region_name})', fontweight='bold', fontsize=14)
    ax1.set_xlabel('Number of Graduates')
    ax1.legend(title='Gender')
    ax1.grid(axis='x', alpha=0.3)
    
    # Plot 2: % Female
    ax2 = axes[1]
    pct_data = field_gender['% Female'].head(10)
    colors = ['#FF1493' if x >= 50 else '#1E90FF' for x in pct_data]
    pct_data.plot(kind='barh', ax=ax2, color=colors, width=0.7)
    ax2.axvline(x=50, color='black', linestyle='--', linewidth=2, label='50% (parity)')
    ax2.set_title(f'% Female by Field ({region_name})', fontweight='bold', fontsize=14)
    ax2.set_xlabel('% Female')
    ax2.set_xlim(0, 100)
    ax2.legend()
    ax2.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / 'field_of_study_gender_analysis.png'
    plt.savefig(output_path, dpi=250, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_path}")
    plt.close()
    
    # Important note
    print("\n" + "="*80)
    print("NOTE ON DATA LIMITATIONS")
    print("="*80)
    print("\nThis table (37100168) only has high-level field groupings (STEM vs BHASE).")
    print("For detailed field-by-field analysis, we need a different table.")
    print("\nThe STEM vs BHASE comparison shows:")
    print("• STEM = Science, Technology, Engineering, Mathematics")
    print("• BHASE = Business, Humanities, Arts, Social Science, Education")
    print("\nSTEM fields typically lead to higher-paying careers ($70-90K)")
    print("BHASE fields typically lead to lower-paying careers ($45-65K)")
    print(f"\nWomen in STEM: {stem_fields.iloc[0]['% Female']:.1f}% (underrepresented)" if not stem_fields.empty else "")
    print(f"Women in BHASE: {bhase_fields.iloc[0]['% Female']:.1f}%" if not bhase_fields.empty else "")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description='Analyze gender disparities in field of study')
    ap.add_argument('--csv', required=True, help='Path to StatCan CSV file')
    ap.add_argument('--output_dir', default='outputs', help='Output directory')
    args = ap.parse_args()
    
    analyze_field_of_study(args.csv, args.output_dir)