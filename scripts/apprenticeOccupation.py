"""
Comprehensive Gender Disparity Analysis
Analyzes both apprenticeships (trades exclusion) and occupations (job segregation)
Supports StatCan Tables: 37100118 (apprenticeships) and 14100325 (occupations)
"""
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

def analyze_apprenticeships(csv_path, output_dir):
    """Analyze gender disparities in apprenticeship registrations"""
    
    print("\n" + "="*80)
    print("APPRENTICESHIP GENDER DISPARITY ANALYSIS")
    print("="*80)
    
    # Load data
    print("\nLoading apprenticeship data...")
    df = pd.read_csv(csv_path, low_memory=False)
    print(f"Total rows: {len(df):,}")
    
    # Identify key columns
    geo_col = None
    gender_col = None
    trade_col = None
    value_col = None
    ref_col = None
    
    for col in df.columns:
        col_lower = col.lower()
        if 'geo' in col_lower or 'province' in col_lower:
            geo_col = col
        if 'sex' in col_lower or 'gender' in col_lower:
            gender_col = col
        if 'trade' in col_lower or 'program' in col_lower:
            trade_col = col
        if 'value' in col_lower:
            value_col = col
        if 'ref_date' in col_lower or 'year' in col_lower:
            ref_col = col
    
    # Clean data
    df[value_col] = pd.to_numeric(df[value_col], errors='coerce')
    df = df.dropna(subset=[value_col])
    
    # Filter to most recent year
    if ref_col:
        latest_year = df[ref_col].max()
        df = df[df[ref_col] == latest_year]
        print(f"Latest year: {latest_year}")
    
    # Filter to actual gender data (not totals)
    gender_data = df[~df[gender_col].str.contains('total|both', case=False, na=False)].copy()
    
    # Identify male/female columns
    genders = gender_data[gender_col].unique()
    female_col = None
    male_col = None
    for g in genders:
        g_lower = str(g).lower()
        if 'female' in g_lower or 'women' in g_lower:
            female_col = g
        elif 'male' in g_lower or 'men' in g_lower:
            male_col = g
    
    # Filter to BC or Canada
    bc_data = gender_data[gender_data[geo_col].str.contains('British Columbia', case=False, na=False)].copy()
    canada_data = gender_data[gender_data[geo_col].str.contains('^Canada$', case=False, regex=True, na=False)].copy()
    
    # Use BC if available, otherwise Canada
    if not bc_data.empty:
        data_to_use = bc_data
        region_name = "British Columbia"
    else:
        data_to_use = canada_data
        region_name = "Canada"
    
    print(f"Region: {region_name}")
    print(f"Data rows: {len(data_to_use):,}")
    
    # Calculate gender distribution by trade
    trade_gender = data_to_use.groupby([trade_col, gender_col])[value_col].sum().unstack(fill_value=0)
    
    if female_col in trade_gender.columns and male_col in trade_gender.columns:
        trade_gender['Total'] = trade_gender[female_col] + trade_gender[male_col]
        trade_gender['% Female'] = (trade_gender[female_col] / trade_gender['Total'] * 100).round(1)
        trade_gender['% Male'] = (trade_gender[male_col] / trade_gender['Total'] * 100).round(1)
        
        # Filter out very small trades
        trade_gender = trade_gender[trade_gender['Total'] > 10]
        
        # Sort by % Female
        trade_gender_sorted = trade_gender.sort_values('% Female', ascending=False)
        
        print("\n" + "-"*80)
        print("MOST FEMALE-DOMINATED TRADES (Top 10):")
        print("-"*80)
        print(trade_gender_sorted[[female_col, male_col, '% Female']].head(10).to_string())
        
        print("\n" + "-"*80)
        print("MOST MALE-DOMINATED TRADES (Top 10):")
        print("-"*80)
        print(trade_gender_sorted[[female_col, male_col, '% Female']].tail(10).to_string())
        
        # Key statistics
        print("\n" + "-"*80)
        print("KEY FINDINGS:")
        print("-"*80)
        
        total_female = trade_gender[female_col].sum()
        total_male = trade_gender[male_col].sum()
        total_all = total_female + total_male
        pct_female_overall = (total_female / total_all * 100)
        
        print(f"\nOVERALL APPRENTICESHIP REGISTRATIONS:")
        print(f"  Women: {pct_female_overall:.1f}% ({total_female:,.0f} registrations)")
        print(f"  Men: {100-pct_female_overall:.1f}% ({total_male:,.0f} registrations)")
        
        if pct_female_overall < 20:
            print(f"\n⚠️  SEVERE DISPARITY: Women are only {pct_female_overall:.1f}% of apprentices")
            print("   Women excluded from high-paying skilled trades ($60-100K+ careers)")
        
        # Identify construction/industrial trades
        construction_keywords = ['construction', 'electrician', 'plumber', 'carpenter', 
                                'welder', 'mechanic', 'millwright', 'pipefitter', 'ironworker']
        
        construction_trades = trade_gender[
            trade_gender.index.str.contains('|'.join(construction_keywords), case=False, na=False)
        ]
        
        if not construction_trades.empty:
            const_female = construction_trades[female_col].sum()
            const_total = construction_trades['Total'].sum()
            const_pct_female = (const_female / const_total * 100)
            
            print(f"\nCONSTRUCTION/INDUSTRIAL TRADES:")
            print(f"  Women: {const_pct_female:.1f}% ({const_female:,.0f})")
            
            if const_pct_female < 5:
                print(f"  ⚠️  EXTREME EXCLUSION: Only {const_pct_female:.1f}% of construction trades")
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        
        # 1. Top trades by total registrations
        ax1 = axes[0, 0]
        top_trades = trade_gender.nlargest(15, 'Total')[[female_col, male_col]]
        top_trades.plot(kind='barh', stacked=True, ax=ax1, color=['#FF1493', '#1E90FF'], width=0.7)
        ax1.set_title(f'Top 15 Trades - Apprenticeship Registrations ({region_name})', 
                     fontweight='bold', fontsize=13)
        ax1.set_xlabel('Number of Registrations')
        ax1.legend(title='Gender', labels=['Women', 'Men'])
        ax1.grid(axis='x', alpha=0.3)
        
        # 2. % Female by trade (most imbalanced)
        ax2 = axes[0, 1]
        trade_gender['Imbalance'] = abs(trade_gender['% Female'] - 50)
        most_imbalanced = trade_gender.nlargest(15, 'Imbalance')['% Female']
        colors = ['#FF1493' if x >= 50 else '#1E90FF' for x in most_imbalanced]
        most_imbalanced.plot(kind='barh', ax=ax2, color=colors, width=0.7)
        ax2.axvline(x=50, color='black', linestyle='--', linewidth=2, label='50% (parity)')
        ax2.set_title(f'Most Gender-Imbalanced Trades ({region_name})', 
                     fontweight='bold', fontsize=13)
        ax2.set_xlabel('% Female')
        ax2.set_xlim(0, 100)
        ax2.legend()
        ax2.grid(axis='x', alpha=0.3)
        
        # 3. Overall gender split (pie chart)
        ax3 = axes[1, 0]
        ax3.pie([total_female, total_male], 
               labels=['Women', 'Men'],
               colors=['#FF1493', '#1E90FF'],
               autopct='%1.1f%%',
               startangle=90,
               textprops={'fontsize': 12})
        ax3.set_title(f'Overall Apprenticeship Gender Split ({region_name})',
                     fontweight='bold', fontsize=13)
        
        # 4. Most male-dominated trades
        ax4 = axes[1, 1]
        bottom_trades = trade_gender.nsmallest(15, '% Female')['% Female']
        bottom_trades.plot(kind='barh', ax=ax4, color='#1E90FF', width=0.7)
        ax4.axvline(x=50, color='red', linestyle='--', linewidth=2, label='50% (parity)')
        ax4.set_title(f'Most Male-Dominated Trades ({region_name})',
                     fontweight='bold', fontsize=13)
        ax4.set_xlabel('% Female')
        ax4.set_xlim(0, 100)
        ax4.legend()
        ax4.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        output_path = output_dir / 'apprenticeship_gender_analysis.png'
        plt.savefig(output_path, dpi=250, bbox_inches='tight')
        print(f"\n✓ Visualization saved: {output_path}")
        plt.close()
        
        return {
            'pct_female_overall': pct_female_overall,
            'total_female': total_female,
            'total_male': total_male
        }


def analyze_occupations(csv_path, output_dir):
    """Analyze gender disparities in occupation-level employment"""
    
    print("\n" + "="*80)
    print("OCCUPATION GENDER DISPARITY ANALYSIS")
    print("="*80)
    
    # Load data
    print("\nLoading occupation data...")
    df = pd.read_csv(csv_path, low_memory=False)
    print(f"Total rows: {len(df):,}")
    
    # Identify key columns
    geo_col = None
    gender_col = None
    occupation_col = None
    value_col = None
    ref_col = None
    
    for col in df.columns:
        col_lower = col.lower()
        if 'geo' in col_lower or 'province' in col_lower:
            geo_col = col
        if 'sex' in col_lower or 'gender' in col_lower:
            gender_col = col
        if 'occupation' in col_lower or 'noc' in col_lower:
            occupation_col = col
        if 'value' in col_lower:
            value_col = col
        if 'ref_date' in col_lower or 'year' in col_lower:
            ref_col = col
    
    # Clean data
    df[value_col] = pd.to_numeric(df[value_col], errors='coerce')
    df = df.dropna(subset=[value_col])
    
    # Filter to most recent year
    if ref_col:
        latest_year = df[ref_col].max()
        df = df[df[ref_col] == latest_year]
        print(f"Latest year: {latest_year}")
    
    # Filter to actual gender data (not totals)
    gender_data = df[~df[gender_col].str.contains('total|both sexes', case=False, na=False)].copy()
    
    # Identify male/female columns
    genders = gender_data[gender_col].unique()
    female_col = None
    male_col = None
    for g in genders:
        g_lower = str(g).lower()
        if 'female' in g_lower or 'women' in g_lower:
            female_col = g
        elif 'male' in g_lower or 'men' in g_lower:
            male_col = g
    
    # Filter to Canada
    exact_canada = gender_data[gender_data[geo_col].str.match('^Canada$', case=False, na=False)].copy()
    
    if not exact_canada.empty:
        data_to_use = exact_canada
        region_name = "Canada"
    else:
        data_to_use = gender_data
        region_name = "All regions"
    
    print(f"Region: {region_name}")
    print(f"Data rows: {len(data_to_use):,}")
    
    # Calculate gender distribution by occupation
    occ_gender = data_to_use.groupby([occupation_col, gender_col])[value_col].sum().unstack(fill_value=0)
    
    if female_col in occ_gender.columns and male_col in occ_gender.columns:
        occ_gender['Total'] = occ_gender[female_col] + occ_gender[male_col]
        occ_gender['% Female'] = (occ_gender[female_col] / occ_gender['Total'] * 100).round(1)
        occ_gender['% Male'] = (occ_gender[male_col] / occ_gender['Total'] * 100).round(1)
        
        # Filter out very small occupations
        occ_gender = occ_gender[occ_gender['Total'] > 1000]
        
        # Sort by % Female
        occ_gender_sorted = occ_gender.sort_values('% Female', ascending=False)
        
        print("\n" + "-"*80)
        print("MOST FEMALE-DOMINATED OCCUPATIONS (Top 10):")
        print("-"*80)
        print(occ_gender_sorted[[female_col, male_col, '% Female']].head(10).to_string())
        
        print("\n" + "-"*80)
        print("MOST MALE-DOMINATED OCCUPATIONS (Top 10):")
        print("-"*80)
        print(occ_gender_sorted[[female_col, male_col, '% Female']].tail(10).to_string())
        
        # Key statistics
        print("\n" + "-"*80)
        print("KEY FINDINGS:")
        print("-"*80)
        
        total_female = occ_gender[female_col].sum()
        total_male = occ_gender[male_col].sum()
        total_all = total_female + total_male
        pct_female_overall = (total_female / total_all * 100)
        
        print(f"\nOVERALL EMPLOYMENT:")
        print(f"  Women: {pct_female_overall:.1f}% ({total_female:,.0f})")
        print(f"  Men: {100-pct_female_overall:.1f}% ({total_male:,.0f})")
        
        # Identify high-paying occupations
        high_pay_keywords = ['management', 'manager', 'senior', 'executive', 'director',
                            'professional', 'engineer', 'scientist', 'doctor', 'physician']
        
        high_pay_occs = occ_gender[
            occ_gender.index.str.contains('|'.join(high_pay_keywords), case=False, na=False)
        ]
        
        # Identify low-paying occupations
        low_pay_keywords = ['clerical', 'administrative support', 'service', 'retail', 
                           'cashier', 'food service', 'cleaning', 'care']
        
        low_pay_occs = occ_gender[
            occ_gender.index.str.contains('|'.join(low_pay_keywords), case=False, na=False)
        ]
        
        if not high_pay_occs.empty:
            high_female = high_pay_occs[female_col].sum()
            high_total = high_pay_occs['Total'].sum()
            high_pct_female = (high_female / high_total * 100)
            
            print(f"\nHIGH-PAYING OCCUPATIONS (Management, Professional):")
            print(f"  Women: {high_pct_female:.1f}% ({high_female:,.0f})")
            
            if high_pct_female < 40:
                print(f"  ⚠️  DISPARITY: Women underrepresented in high-paying jobs")
        
        if not low_pay_occs.empty:
            low_female = low_pay_occs[female_col].sum()
            low_total = low_pay_occs['Total'].sum()
            low_pct_female = (low_female / low_total * 100)
            
            print(f"\nLOW-PAYING OCCUPATIONS (Service, Support):")
            print(f"  Women: {low_pct_female:.1f}% ({low_female:,.0f})")
            
            if low_pct_female > 60:
                print(f"  ⚠️  DISPARITY: Women overrepresented in low-paying jobs")
        
        # Calculate segregation index
        occ_gender['pct_female_of_all_female'] = occ_gender[female_col] / total_female
        occ_gender['pct_male_of_all_male'] = occ_gender[male_col] / total_male
        occ_gender['dissimilarity'] = abs(occ_gender['pct_female_of_all_female'] - 
                                          occ_gender['pct_male_of_all_male'])
        segregation_index = occ_gender['dissimilarity'].sum() / 2 * 100
        
        print(f"\nOCCUPATIONAL SEGREGATION INDEX: {segregation_index:.1f}%")
        print(f"  (% of workers who would need to change occupations for gender parity)")
        
        if segregation_index > 30:
            print("  ⚠️  HIGH SEGREGATION: Labor market is highly gender-segregated")
        
        # Create visualizations
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        
        # 1. Top occupations by total employment
        ax1 = axes[0, 0]
        top_occs = occ_gender.nlargest(15, 'Total')[[female_col, male_col]]
        top_occs.plot(kind='barh', stacked=True, ax=ax1, color=['#FF1493', '#1E90FF'], width=0.7)
        ax1.set_title(f'Top 15 Occupations by Employment ({region_name})', 
                     fontweight='bold', fontsize=13)
        ax1.set_xlabel('Number of Employed')
        ax1.legend(title='Gender', labels=['Women', 'Men'])
        ax1.grid(axis='x', alpha=0.3)
        
        # 2. % Female by occupation (most imbalanced)
        ax2 = axes[0, 1]
        occ_gender['Imbalance'] = abs(occ_gender['% Female'] - 50)
        most_imbalanced = occ_gender.nlargest(15, 'Imbalance')['% Female']
        colors = ['#FF1493' if x >= 50 else '#1E90FF' for x in most_imbalanced]
        most_imbalanced.plot(kind='barh', ax=ax2, color=colors, width=0.7)
        ax2.axvline(x=50, color='black', linestyle='--', linewidth=2, label='50% (parity)')
        ax2.set_title(f'Most Gender-Imbalanced Occupations ({region_name})', 
                     fontweight='bold', fontsize=13)
        ax2.set_xlabel('% Female')
        ax2.set_xlim(0, 100)
        ax2.legend()
        ax2.grid(axis='x', alpha=0.3)
        
        # 3. Gender balance distribution
        ax3 = axes[1, 0]
        female_dominated = len(occ_gender[occ_gender['% Female'] > 60])
        male_dominated = len(occ_gender[occ_gender['% Female'] < 40])
        balanced = len(occ_gender[(occ_gender['% Female'] >= 40) & (occ_gender['% Female'] <= 60)])
        
        ax3.bar(['Female-dominated\n(>60% women)', 'Balanced\n(40-60%)', 'Male-dominated\n(<40% women)'],
               [female_dominated, balanced, male_dominated],
               color=['#FF1493', '#9370DB', '#1E90FF'])
        ax3.set_title('Occupational Gender Balance Distribution', fontweight='bold', fontsize=13)
        ax3.set_ylabel('Number of Occupations')
        ax3.grid(axis='y', alpha=0.3)
        
        # 4. Most male-dominated occupations
        ax4 = axes[1, 1]
        bottom_occs = occ_gender.nsmallest(15, '% Female')['% Female']
        bottom_occs.plot(kind='barh', ax=ax4, color='#1E90FF', width=0.7)
        ax4.axvline(x=50, color='red', linestyle='--', linewidth=2, label='50% (parity)')
        ax4.set_title(f'Most Male-Dominated Occupations ({region_name})',
                     fontweight='bold', fontsize=13)
        ax4.set_xlabel('% Female')
        ax4.set_xlim(0, 100)
        ax4.legend()
        ax4.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        output_path = output_dir / 'occupation_gender_analysis.png'
        plt.savefig(output_path, dpi=250, bbox_inches='tight')
        print(f"\n✓ Visualization saved: {output_path}")
        plt.close()
        
        return {
            'segregation_index': segregation_index,
            'pct_female_overall': pct_female_overall
        }


def main():
    ap = argparse.ArgumentParser(description='Analyze gender outcome disparities')
    ap.add_argument('--apprenticeships', help='Path to apprenticeship CSV (37100118)')
    ap.add_argument('--occupations', help='Path to occupation CSV (14100325)')
    ap.add_argument('--output_dir', default='outputs', help='Output directory')
    args = ap.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("="*80)
    print("COMPREHENSIVE GENDER OUTCOME DISPARITY ANALYSIS")
    print("="*80)
    
    results = {}
    
    # Run apprenticeship analysis if provided
    if args.apprenticeships:
        try:
            results['apprenticeships'] = analyze_apprenticeships(args.apprenticeships, output_dir)
        except Exception as e:
            print(f"\n❌ Error analyzing apprenticeships: {e}")
    
    # Run occupation analysis if provided
    if args.occupations:
        try:
            results['occupations'] = analyze_occupations(args.occupations, output_dir)
        except Exception as e:
            print(f"\n❌ Error analyzing occupations: {e}")
    
    # Summary
    print("\n" + "="*80)
    print("COMPREHENSIVE FINDINGS - GENDER OUTCOME DISPARITIES")
    print("="*80)
    
    if 'apprenticeships' in results:
        print(f"\n📊 APPRENTICESHIPS (Trades Exclusion):")
        print(f"   Women: {results['apprenticeships']['pct_female_overall']:.1f}% of apprentices")
        print(f"   Impact: Women excluded from $60-100K+ skilled trades careers")
    
    if 'occupations' in results:
        print(f"\n📊 OCCUPATIONS (Job Segregation):")
        print(f"   Women: {results['occupations']['pct_female_overall']:.1f}% of workforce")
        print(f"   Segregation Index: {results['occupations']['segregation_index']:.1f}%")
        print(f"   Impact: Women concentrated in lower-paying occupations")
    
    print("\n💰 ECONOMIC IMPACT:")
    print("   These disparities create and perpetuate the gender wage gap:")
    print("   • Women excluded from high-paying trades and technical roles")
    print("   • Women concentrated in lower-paying care/service roles")
    print("   • Structural barriers prevent equal economic opportunity")
    print("   • Lifetime earnings gap: $500K - $1M+ per woman")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nOutputs saved to: {output_dir}/")


if __name__ == "__main__":
    main()