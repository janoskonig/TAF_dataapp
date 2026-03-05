#!/usr/bin/env python3
"""
Script to check for statistically significant results in the cross-sectional analyses.
"""

import sys
import os
from dotenv import load_dotenv
import mysql.connector

# Add the current directory to path to import from main
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Load environment variables
load_dotenv(dotenv_path=".env")

# Database connection
host = os.getenv("DB_HOST")
port = os.getenv("DB_PORT")
user = os.getenv("DB_USER")
password = os.getenv("DB_PASSWORD")
database = os.getenv("DB_NAME")

def create_db_connection():
    return mysql.connector.connect(
        host=host,
        port=int(port),
        user=user,
        password=password,
        database=database
    )

def get_db_cursor():
    db = create_db_connection()
    return db.cursor(), db

if __name__ == "__main__":
    try:
        # Import the analysis function from main
        from main import perform_cross_sectional_analysis
        
        cursor, db = get_db_cursor()
        
        print("Running cross-sectional analyses...")
        print("=" * 60)
        
        results = perform_cross_sectional_analysis(cursor)
        
        # Check for significant results
        significant_results = []
        non_significant_results = []
        insufficient_data = []
        errors = []
        
        for hyp_id in ['H1a', 'H1b', 'H1c', 'H2a', 'H2b', 'H2c', 'H3a', 'H3b', 'H4a', 'H4b', 'H5', 'H6a', 'H6b', 'H7']:
            if hyp_id in results:
                hyp = results[hyp_id]
                status = hyp.get('status', 'unknown')
                
                if status == 'success':
                    p_value = hyp.get('p_value')
                    f_p_value = hyp.get('f_p_value')  # For regression models
                    
                    # Check significance
                    is_significant = False
                    if p_value is not None and p_value < 0.05:
                        is_significant = True
                    elif f_p_value is not None and f_p_value < 0.05:
                        is_significant = True
                    
                    if is_significant:
                        significant_results.append((hyp_id, hyp))
                    else:
                        non_significant_results.append((hyp_id, hyp))
                        
                elif status == 'insufficient_data':
                    insufficient_data.append((hyp_id, hyp))
                elif status == 'error':
                    errors.append((hyp_id, hyp))
        
        # Print results
        print(f"\n📊 STATISTICALLY SIGNIFICANT RESULTS (p < 0.05):")
        print("=" * 60)
        if significant_results:
            for hyp_id, hyp in significant_results:
                p_val = hyp.get('p_value') or hyp.get('f_p_value', 'N/A')
                test_name = hyp.get('test_name', 'N/A')
                n = hyp.get('n', 'N/A')
                
                if 'correlation' in hyp:
                    corr = hyp.get('correlation', 'N/A')
                    print(f"✅ {hyp_id}: {test_name}")
                    print(f"   Correlation: {corr:.3f}, p = {p_val:.4f}, n = {n}")
                elif 'f_p_value' in hyp:
                    r_sq = hyp.get('r_squared', 'N/A')
                    print(f"✅ {hyp_id}: {test_name}")
                    print(f"   R² = {r_sq:.3f}, F-test p = {p_val:.4f}, n = {n}")
                else:
                    stat = hyp.get('statistic', 'N/A')
                    print(f"✅ {hyp_id}: {test_name}")
                    print(f"   Statistic: {stat:.3f}, p = {p_val:.4f}, n = {n}")
                print()
        else:
            print("   No statistically significant results found (p < 0.05)")
        
        print(f"\n📉 NON-SIGNIFICANT RESULTS (p ≥ 0.05):")
        print("=" * 60)
        if non_significant_results:
            for hyp_id, hyp in non_significant_results:
                p_val = hyp.get('p_value') or hyp.get('f_p_value', 'N/A')
                test_name = hyp.get('test_name', 'N/A')
                n = hyp.get('n', 'N/A')
                print(f"   {hyp_id}: {test_name}, p = {p_val:.4f}, n = {n}")
        else:
            print("   (All successful analyses were significant)")
        
        print(f"\n⚠️  INSUFFICIENT DATA:")
        print("=" * 60)
        if insufficient_data:
            for hyp_id, hyp in insufficient_data:
                message = hyp.get('message', 'N/A')
                n = hyp.get('n', 'N/A')
                print(f"   {hyp_id}: {message} (n = {n})")
        else:
            print("   None")
        
        print(f"\n❌ ERRORS:")
        print("=" * 60)
        if errors:
            for hyp_id, hyp in errors:
                message = hyp.get('message', 'N/A')
                print(f"   {hyp_id}: {message}")
        else:
            print("   None")
        
        print("\n" + "=" * 60)
        print(f"Summary: {len(significant_results)} significant, {len(non_significant_results)} non-significant, {len(insufficient_data)} insufficient data, {len(errors)} errors")
        
        cursor.close()
        db.close()
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


