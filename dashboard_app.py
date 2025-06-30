import streamlit as st
import pandas as pd
import plotly.express as px
import glob

# 1. Page config must be first!
st.set_page_config(page_title="Farmer Data Dashboard", layout="wide")

# Only call st.login("google") on Streamlit Cloud
try:
    st.login("google")
except Exception:
    pass  # If running locally, ignore

if not hasattr(st.user, "is_logged_in") or not st.user.is_logged_in:
    st.stop()

if not getattr(st.user, "email", "").endswith("@agrivijay.com"):
    st.error("Access restricted: Only AgriVijay users with @agrivijay.com email can view this dashboard.")
    st.stop()

st.success(f"Welcome, {getattr(st.user, 'email', 'user')}!")

# --- Load farmer data ---
@st.cache_data
def load_farmer_data():
    detailed = pd.read_csv('data/detailed_transactions.csv')
    sales = pd.read_csv('data/product_sales_metrics.csv', header=2)
    detailed.columns = detailed.columns.str.strip()
    sales.columns = sales.columns.str.strip()
    return detailed, sales

detailed, sales = load_farmer_data()

# --- Fix column types ---
if 'No of Units' in sales.columns:
    sales['No of Units'] = pd.to_numeric(sales['No of Units'], errors='coerce')
if 'Revenue Contribution in INR' in sales.columns:
    sales['Revenue Contribution in INR'] = pd.to_numeric(sales['Revenue Contribution in INR'], errors='coerce')

if 'Invoice Amount(in Rs.)' in detailed.columns:
    detailed['Invoice Amount(in Rs.)'] = pd.to_numeric(detailed['Invoice Amount(in Rs.)'], errors='coerce')
if 'GHG / CO2 Emissions Abated / Year' in detailed.columns:
    detailed['GHG / CO2 Emissions Abated / Year'] = pd.to_numeric(detailed['GHG / CO2 Emissions Abated / Year'], errors='coerce')

# --- Sidebar filters for farmer data ---
st.sidebar.header("Filters")
detailed['Date'] = pd.to_datetime(detailed['Date'], errors='coerce')
detailed['Year'] = detailed['Date'].dt.year
years = sorted(detailed['Year'].dropna().unique())
selected_year = st.sidebar.selectbox("Select Year", options=["All"] + years, index=0)
product_categories = sorted(detailed['Product Category'].dropna().unique())
selected_category = st.sidebar.selectbox("Select Product Category", options=["All"] + product_categories, index=0)
states = sorted(detailed['State/Branch'].dropna().unique())
selected_state = st.sidebar.selectbox("Select State/Branch", options=["All"] + states, index=0)

# Apply filters
filtered = detailed.copy()
if selected_year != "All":
    filtered = filtered[filtered['Year'] == selected_year]
if selected_category != "All":
    filtered = filtered[filtered['Product Category'] == selected_category]
if selected_state != "All":
    filtered = filtered[filtered['State/Branch'] == selected_state]

# --- Farmer Data KPIs ---
st.title("Farmer Data Dashboard")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Revenue (₹)", f"{filtered['Invoice Amount(in Rs.)'].sum():,.0f}")
col2.metric("Units Sold", f"{sales['No of Units'].sum():.0f}")
col3.metric("GHG Abated (Yearly, tCO2e)", f"{filtered['GHG / CO2 Emissions Abated / Year'].sum():,.2f}")
col4.metric("Unique Customers", f"{filtered['Customer Name'].nunique()}")

# --- Product Sales Metrics Overview ---
st.header("Product Sales Metrics Overview")

if 'Product Name' in sales.columns and 'No of Units' in sales.columns:
    fig_units = px.bar(
        sales,
        x='Product Name',
        y='No of Units',
        color='Category',
        title='Units Sold per Product'
    )
    st.plotly_chart(fig_units, use_container_width=True)

if 'Product Name' in sales.columns and 'Revenue Contribution in INR' in sales.columns:
    fig_revenue = px.bar(
        sales,
        x='Product Name',
        y='Revenue Contribution in INR',
        color='Category',
        title='Revenue per Product'
    )
    st.plotly_chart(fig_revenue, use_container_width=True)

# --- Year-over-Year Growth ---
st.header("Year-over-Year Growth Analysis")
yearly = filtered.groupby('Year').agg({'Invoice Amount(in Rs.)':'sum', 'GHG / CO2 Emissions Abated / Year':'sum'}).reset_index()
if not yearly.empty:
    fig_yoy = px.bar(yearly, x='Year', y=['Invoice Amount(in Rs.)', 'GHG / CO2 Emissions Abated / Year'],
                     barmode='group', title='Revenue & GHG Abatement by Year')
    st.plotly_chart(fig_yoy, use_container_width=True)

# --- Detailed Transactions Analysis ---
st.header("Detailed Transactions Analysis")

# Sales by State
if 'State/Branch' in filtered.columns and 'Product' in filtered.columns:
    state_sales = filtered.groupby('State/Branch').size().reset_index(name='Count')
    fig_state = px.bar(
        state_sales,
        x='State/Branch',
        y='Count',
        title='Number of Transactions by State/Branch'
    )
    st.plotly_chart(fig_state, use_container_width=True)

# Sales by Product Category
if 'Product Category' in filtered.columns:
    prod_cat_sales = filtered['Product Category'].value_counts().reset_index()
    prod_cat_sales.columns = ['Product Category', 'Count']
    fig_cat = px.pie(
        prod_cat_sales,
        names='Product Category',
        values='Count',
        title='Sales by Product Category'
    )
    st.plotly_chart(fig_cat, use_container_width=True)

# Sales over Time
if 'Date' in filtered.columns:
    sales_over_time = filtered.groupby(filtered['Date'].dt.to_period('M')).size().reset_index(name='Count')
    sales_over_time['Date'] = sales_over_time['Date'].astype(str)
    fig_time = px.line(
        sales_over_time,
        x='Date',
        y='Count',
        title='Sales Trend Over Time'
    )
    st.plotly_chart(fig_time, use_container_width=True)

# --- Customer Segmentation ---
st.header("Customer Segmentation & Top Customers")
if 'Customer Name' in filtered.columns:
    customer_counts = filtered['Customer Name'].value_counts().reset_index()
    customer_counts.columns = ['Customer Name', 'Purchases']
    fig_customers = px.bar(customer_counts.head(10), x='Customer Name', y='Purchases', title='Top 10 Customers by Purchases')
    st.plotly_chart(fig_customers, use_container_width=True)
    st.metric("Repeat Customers", customer_counts[customer_counts['Purchases'] > 1].shape[0])

# --- Product Impact Leaderboard ---
st.header("Product Impact Leaderboard")
if 'Product' in filtered.columns and 'GHG / CO2 Emissions Abated / Year' in filtered.columns:
    impact = filtered.groupby('Product').agg({'GHG / CO2 Emissions Abated / Year':'sum'}).reset_index().sort_values(
        by='GHG / CO2 Emissions Abated / Year', ascending=False)
    st.dataframe(impact.head(10))

# --- Revenue & GHG per Category ---
st.header("Revenue & GHG Abatement per Category")
if 'Product Category' in filtered.columns and 'Invoice Amount(in Rs.)' in filtered.columns and 'GHG / CO2 Emissions Abated / Year' in filtered.columns:
    cat_metrics = filtered.groupby('Product Category').agg({'Invoice Amount(in Rs.)':'sum', 'GHG / CO2 Emissions Abated / Year':'sum'}).reset_index()
    fig_cat = px.bar(cat_metrics, x='Product Category', y=['Invoice Amount(in Rs.)', 'GHG / CO2 Emissions Abated / Year'],
                     barmode='group', title='Revenue & GHG Abatement per Category')
    st.plotly_chart(fig_cat, use_container_width=True)

# --- Monthly Revenue Trend with Simple Forecast (Rolling Mean) ---
st.header("Monthly Revenue Trend & Forecast")
if 'Date' in filtered.columns and 'Invoice Amount(in Rs.)' in filtered.columns:
    monthly_revenue = filtered.groupby(filtered['Date'].dt.to_period('M'))['Invoice Amount(in Rs.)'].sum().reset_index()
    monthly_revenue['Date'] = monthly_revenue['Date'].astype(str)
    monthly_revenue['Forecast'] = monthly_revenue['Invoice Amount(in Rs.)'].rolling(window=3, min_periods=1).mean()
    fig_monthly_revenue = px.line(
        monthly_revenue,
        x='Date',
        y=['Invoice Amount(in Rs.)', 'Forecast'],
        title='Monthly Revenue Trend with Forecast'
    )
    st.plotly_chart(fig_monthly_revenue, use_container_width=True)

# --- Additional Insights ---
st.header("Additional Insights")

# Average Invoice Amount per Product Category
if 'Product Category' in filtered.columns and 'Invoice Amount(in Rs.)' in filtered.columns:
    avg_invoice = filtered.groupby('Product Category')['Invoice Amount(in Rs.)'].mean().reset_index()
    st.subheader("Average Invoice Amount per Product Category")
    st.dataframe(avg_invoice)

# Top 5 Customers by Total Invoice Amount
if 'Customer Name' in filtered.columns and 'Invoice Amount(in Rs.)' in filtered.columns:
    top_customers = filtered.groupby('Customer Name')['Invoice Amount(in Rs.)'].sum().reset_index().sort_values(
        by='Invoice Amount(in Rs.)', ascending=False).head(5)
    st.subheader("Top 5 Customers by Total Invoice Amount")
    st.dataframe(top_customers)

# Product Category Contribution to Total Revenue
if 'Product Category' in filtered.columns and 'Invoice Amount(in Rs.)' in filtered.columns:
    revenue_by_category = filtered.groupby('Product Category')['Invoice Amount(in Rs.)'].sum().reset_index()
    st.subheader("Revenue by Product Category")
    st.dataframe(revenue_by_category)
    fig_revenue_cat = px.pie(
        revenue_by_category,
        names='Product Category',
        values='Invoice Amount(in Rs.)',
        title='Revenue Share by Product Category'
    )
    st.plotly_chart(fig_revenue_cat, use_container_width=True)

# Average GHG/CO2 Emissions Abated per Product Category
if 'Product Category' in filtered.columns and 'GHG / CO2 Emissions Abated / Year' in filtered.columns:
    avg_ghg = filtered.groupby('Product Category')['GHG / CO2 Emissions Abated / Year'].mean().reset_index()
    st.subheader("Average GHG/CO2 Emissions Abated per Product Category")
    st.dataframe(avg_ghg)
    fig_ghg = px.bar(
        avg_ghg,
        x='Product Category',
        y='GHG / CO2 Emissions Abated / Year',
        title='Average GHG/CO2 Emissions Abated per Product Category'
    )
    st.plotly_chart(fig_ghg, use_container_width=True)

# --- Customer Business Records Lookup ---
st.header("🔎 Customer Business Records Lookup")

customer_names = detailed['Customer Name'].dropna().unique()
customer_names = sorted(customer_names)
selected_customer = st.selectbox("Select a customer to view all their records:", customer_names)
search_text = st.text_input("Or search by customer name (partial match):", key="customer_search")

if search_text:
    customer_records = detailed[detailed['Customer Name'].str.contains(search_text, case=False, na=False)]
    st.write(f"Showing results for search: **{search_text}**")
elif selected_customer:
    customer_records = detailed[detailed['Customer Name'] == selected_customer]
    st.write(f"Showing all records for: **{selected_customer}**")
else:
    customer_records = pd.DataFrame()

if not customer_records.empty:
    st.dataframe(customer_records)
    st.success(f"Found {len(customer_records)} record(s) for this customer.")
else:
    st.info("No records found for the given customer.")

# --- Load franchise/partner sheets ---
@st.cache_data
def load_and_clean_franchise_data():
    """Load and clean all franchise data files with robust error handling"""
    all_dfs = []
    
    # Define file patterns to search for
    file_patterns = [
        "data/franchise_*.csv",
        "data/paste-*.txt",
        "data/franchise_*.txt"
    ]
    
    files = []
    for pattern in file_patterns:
        files.extend(glob.glob(pattern))
    
    files = sorted(set(files))  # Remove duplicates and sort
    
    if not files:
        st.warning("No franchise data files found. Please check your data directory.")
        return pd.DataFrame()
    
    for f in files:
        try:
            # Try different separators and encodings
            df = None
            for sep in ['\t', ',', ';']:
                try:
                    df = pd.read_csv(f, sep=sep, dtype=str, engine='python', encoding='utf-8')
                    if df.shape[1] > 1:  # If we got multiple columns, it worked
                        break
                except:
                    continue
            
            if df is None or df.shape[1] <= 1:
                # Try with different encoding
                try:
                    df = pd.read_csv(f, sep='\t', dtype=str, engine='python', encoding='latin-1')
                except:
                    st.warning(f"Could not read file: {f}")
                    continue
            
            # Clean and standardize column names
            df.columns = df.columns.str.strip().str.replace(r'\s+', ' ', regex=True)
            
            # Clean all string values
            for col in df.columns:
                if df[col].dtype == object:
                    df[col] = df[col].astype(str).str.strip()
                    df[col] = df[col].replace(['nan', 'NaN', 'None', ''], pd.NA)
            
            # Standardize key column names
            column_mapping = {
                'State ': 'State',
                'District ': 'District', 
                'Franchise Owner Name ': 'Franchise Owner Name',
                'Franchise Name ': 'Franchise Name',
                'Revenue Collected till Date (in Rs.) ': 'Revenue Collected till Date (in Rs.)',
                'Franchise Amount Paid (in Rs.) ': 'Franchise Amount Paid (in Rs.)'
            }
            
            df = df.rename(columns=column_mapping)
            
            # Extract year from various date columns
            year_extracted = False
            for date_col in ['Onboarding Month', 'Date of Conversion', 'Date']:
                if date_col in df.columns and not year_extracted:
                    if date_col == 'Onboarding Month':
                        df['Year'] = df[date_col].str.extract(r"(\d{2,4})")
                        df['Year'] = df['Year'].apply(
                            lambda x: int("20" + x[-2:]) if pd.notnull(x) and len(str(x)) == 2 
                            else (int(x) if pd.notnull(x) and str(x).isdigit() else None)
                        )
                    else:
                        df['Year'] = pd.to_datetime(df[date_col], errors='coerce').dt.year
                    year_extracted = True
            
            # Add source file info
            df['Source_File'] = f.split('/')[-1]
            
            all_dfs.append(df)
            
        except Exception as e:
            st.error(f"Error processing file {f}: {str(e)}")
            continue
    
    if not all_dfs:
        st.error("No franchise data could be loaded successfully.")
        return pd.DataFrame()
    
    # Concatenate all dataframes
    df_all = pd.concat(all_dfs, ignore_index=True, sort=False)
    
    # Final cleanup
    df_all = df_all.dropna(how='all')  # Remove completely empty rows
    
    return df_all

# Load data
df = load_and_clean_franchise_data()

if df.empty:
    st.stop()

# Display data info
st.sidebar.info(f"📊 Total Records: {len(df)}")

# --- Enhanced Sidebar Filters ---
st.sidebar.header("🔍 Franchise Filters")

# Dynamic filter options (only show available values)
years = sorted([y for y in df['Year'].dropna().unique() if pd.notnull(y)])
states = sorted([s for s in df['State'].dropna().unique() if s not in ['nan', 'None']])
districts = sorted([d for d in df['District'].dropna().unique() if d not in ['nan', 'None']])

# Filter by Data Source
if 'Source_File' in df.columns:
    sources = sorted(df['Source_File'].dropna().unique())
    selected_sources = st.sidebar.multiselect(
        "📁 Filter by Data Source", 
        sources, 
        default=sources,
        key="source_filter"
    )
else:
    selected_sources = []

# Year filter with selectbox for single selection option
year_filter_type = st.sidebar.radio("Year Selection:", ["All Years", "Select Specific Years"], key="year_type")
if year_filter_type == "All Years":
    selected_years = years
else:
    selected_years = st.sidebar.multiselect(
        "📅 Filter by Year", 
        years, 
        default=years,
        key="year_filter"
    )

# State filter with dynamic districts
selected_states = st.sidebar.multiselect(
    "🏛️ Filter by State", 
    states, 
    default=states,
    key="state_filter"
)

# Dynamic districts based on selected states
if selected_states:
    available_districts = sorted([
        d for d in df[df['State'].isin(selected_states)]['District'].dropna().unique() 
        if d not in ['nan', 'None']
    ])
else:
    available_districts = districts

selected_districts = st.sidebar.multiselect(
    "🏘️ Filter by District", 
    available_districts, 
    default=available_districts,
    key="district_filter"
)

# Enhanced search functionality
st.sidebar.subheader("🔎 Search Options")
search_type = st.sidebar.selectbox(
    "Search in:", 
    ["All Fields", "Franchise Name", "Owner Name", "Address", "Contact"],
    key="search_type"
)

search_franchise = st.sidebar.text_input(
    f"Search {search_type.lower()}:", 
    placeholder="Enter search term...",
    key="franchise_search"
)

# Revenue range filter (if revenue column exists)
revenue_cols = [col for col in df.columns if 'Revenue' in col and 'Rs.' in col]
if revenue_cols:
    revenue_col = revenue_cols[0]
    df[revenue_col] = pd.to_numeric(df[revenue_col].str.replace(',', ''), errors='coerce')
    
    if not df[revenue_col].isna().all():
        min_rev, max_rev = int(df[revenue_col].min()), int(df[revenue_col].max())
        revenue_range = st.sidebar.slider(
            "💰 Revenue Range (₹)",
            min_value=min_rev,
            max_value=max_rev,
            value=(min_rev, max_rev),
            key="revenue_filter"
        )
    else:
        revenue_range = None
else:
    revenue_range = None

# Clear filters button
if st.sidebar.button("🔄 Reset All Filters"):
    for key in st.session_state.keys():
        if key.endswith('_filter') or key == 'franchise_search':
            del st.session_state[key]
    st.rerun()

# --- Apply filters ---
filtered_df = df.copy()

# Apply source filter
if selected_sources and 'Source_File' in df.columns:
    filtered_df = filtered_df[filtered_df['Source_File'].isin(selected_sources)]

# Apply year filter
if selected_years:
    filtered_df = filtered_df[filtered_df['Year'].isin(selected_years)]

# Apply state filter
if selected_states:
    filtered_df = filtered_df[filtered_df['State'].isin(selected_states)]

# Apply district filter
if selected_districts:
    filtered_df = filtered_df[filtered_df['District'].isin(selected_districts)]

# Apply revenue filter
if revenue_range and revenue_cols:
    revenue_col = revenue_cols[0]
    filtered_df = filtered_df[
        (filtered_df[revenue_col] >= revenue_range[0]) & 
        (filtered_df[revenue_col] <= revenue_range[1])
    ]

# Apply search filter
if search_franchise:
    if search_type == "All Fields":
        mask = filtered_df.apply(lambda row: search_franchise.lower() in str(row).lower(), axis=1)
    elif search_type == "Franchise Name":
        name_cols = [col for col in filtered_df.columns if 'Franchise Name' in col]
        if name_cols:
            mask = filtered_df[name_cols[0]].str.contains(search_franchise, case=False, na=False)
        else:
            mask = pd.Series([False] * len(filtered_df))
    elif search_type == "Owner Name":
        owner_cols = [col for col in filtered_df.columns if 'Owner Name' in col]
        if owner_cols:
            mask = filtered_df[owner_cols[0]].str.contains(search_franchise, case=False, na=False)
        else:
            mask = pd.Series([False] * len(filtered_df))
    elif search_type == "Address":
        address_cols = [col for col in filtered_df.columns if 'Address' in col]
        if address_cols:
            mask = filtered_df[address_cols[0]].str.contains(search_franchise, case=False, na=False)
        else:
            mask = pd.Series([False] * len(filtered_df))
    elif search_type == "Contact":
        contact_cols = [col for col in filtered_df.columns if 'Contact' in col]
        if contact_cols:
            mask = filtered_df[contact_cols[0]].str.contains(search_franchise, case=False, na=False)
        else:
            mask = pd.Series([False] * len(filtered_df))
    
    filtered_df = filtered_df[mask]

# --- Display Results ---
st.header("🏪 Franchise & Partner Dashboard")

# Summary metrics
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("📊 Total Franchisees", len(filtered_df))
with col2:
    st.metric("🏛️ States", filtered_df['State'].nunique() if 'State' in filtered_df.columns else 0)
with col3:
    st.metric("🏘️ Districts", filtered_df['District'].nunique() if 'District' in filtered_df.columns else 0)
with col4:
    if revenue_cols and not filtered_df[revenue_cols[0]].isna().all():
        total_revenue = filtered_df[revenue_cols[0]].sum()
        st.metric("💰 Total Revenue", f"₹{total_revenue:,.0f}")
    else:
        st.metric("📁 Data Sources", filtered_df['Source_File'].nunique() if 'Source_File' in filtered_df.columns else 0)

# Search results info
if search_franchise:
    st.success(f"🔍 Found {len(filtered_df)} records matching '{search_franchise}' in {search_type.lower()}")

# Display filtered data with enhanced formatting
st.subheader("📋 Filtered Franchise Data")

# Add download button
if not filtered_df.empty:
    csv = filtered_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Filtered Data as CSV",
        data=csv,
        file_name=f"franchise_data_filtered_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv"
    )

# Display the dataframe
st.dataframe(
    filtered_df, 
    use_container_width=True,
    height=400
)

# Show filter summary
with st.expander("📊 Filter Summary", expanded=False):
    st.write(f"**Original Records:** {len(df)}")
    st.write(f"**Filtered Records:** {len(filtered_df)}")
    st.write(f"**Percentage Shown:** {(len(filtered_df)/len(df)*100):.1f}%")
    
    if selected_years != years:
        st.write(f"**Years:** {', '.join(map(str, selected_years))}")
    if selected_states != states:
        st.write(f"**States:** {', '.join(selected_states)}")
    if selected_districts != available_districts:
        st.write(f"**Districts:** {', '.join(selected_districts)}")
    if search_franchise:
        st.write(f"**Search:** '{search_franchise}' in {search_type}")


# --- KPIs ---
col1, col2, col3 = st.columns(3)
col1.metric("Total Franchisees", filtered_df.shape[0])
if 'Revenue Collected till Date (in Rs.)' in filtered_df.columns:
    filtered_df['Revenue Collected till Date (in Rs.)'] = pd.to_numeric(filtered_df['Revenue Collected till Date (in Rs.)'].str.replace(",", ""), errors='coerce')
    col2.metric("Total Revenue (₹)", f"{filtered_df['Revenue Collected till Date (in Rs.)'].sum():,.0f}")
if 'State' in filtered_df.columns:
    col3.metric("States Covered", filtered_df['State'].nunique())

# --- Year-wise trends ---
if 'Year' in filtered_df.columns and 'Revenue Collected till Date (in Rs.)' in filtered_df.columns:
    yearly_rev = filtered_df.groupby('Year')['Revenue Collected till Date (in Rs.)'].sum().reset_index()
    fig = px.line(yearly_rev, x='Year', y='Revenue Collected till Date (in Rs.)', markers=True, title='Year-wise Revenue Trend')
    st.plotly_chart(fig, use_container_width=True)

if 'Year' in filtered_df.columns:
    yearly_count = filtered_df.groupby('Year').size().reset_index(name="Franchisee Count")
    fig2 = px.bar(yearly_count, x='Year', y='Franchisee Count', title='Franchisees Added Each Year')
    st.plotly_chart(fig2, use_container_width=True)

# --- Franchisees by State ---
if 'State' in filtered_df.columns:
    state_counts = filtered_df['State'].value_counts().reset_index()
    state_counts.columns = ['State', 'Count']
    fig3 = px.bar(state_counts, x='State', y='Count', title='Franchisees by State')
    st.plotly_chart(fig3, use_container_width=True)

# --- Franchisees by District ---
if 'District' in filtered_df.columns:
    district_counts = filtered_df['District'].value_counts().reset_index()
    district_counts.columns = ['District', 'Count']
    fig4 = px.bar(district_counts, x='District', y='Count', title='Franchisees by District')
    st.plotly_chart(fig4, use_container_width=True)

# --- Payment Status Pie ---
if 'Remarks (if any)' in filtered_df.columns:
    payment_status = filtered_df['Remarks (if any)'].value_counts().reset_index()
    payment_status.columns = ['Status', 'Count']
    fig5 = px.pie(payment_status, names='Status', values='Count', title='Payment Status Distribution')
    st.plotly_chart(fig5, use_container_width=True)

# --- Franchisee/Owner search results ---
if search_franchise:
    st.success(f"Found {filtered_df.shape[0]} record(s) for search: {search_franchise}")

# @st.cache_data
# def load_yearly_franchise_data():
#     # Adjust the path as needed
#     files = sorted(glob.glob("data/franchise_*.csv"))
#     sheets = {}
#     for f in files:
#         year = f.split("_")[-1].replace(".csv", "").replace("main", "All")
#         df = pd.read_csv(f, sep="\t", dtype=str)
#         df.columns = df.columns.str.strip()
#         # Try to extract year from onboarding/conversion date if not present
#         if "Onboarding Month" in df.columns:
#             df["Year"] = df["Onboarding Month"].str.extract(r"(\d{2,4})")
#             df["Year"] = df["Year"].apply(lambda x: int("20"+x[-2:]) if pd.notnull(x) and len(x)==2 else (int(x) if pd.notnull(x) else None))
#         elif "Date of Conversion" in df.columns:
#             df["Year"] = pd.to_datetime(df["Date of Conversion"], errors="coerce").dt.year
#         elif "Date" in df.columns:
#             df["Year"] = pd.to_datetime(df["Date"], errors="coerce").dt.year
#         sheets[year] = df
#     return sheets

# franchise_sheets = load_yearly_franchise_data()

# # --- Sidebar: select year and filters ---
# years_available = sorted([y for y in franchise_sheets.keys() if y.isdigit() or y == "All"])
# selected_year = st.sidebar.selectbox("Select Year", ["All"] + years_available)
# df = pd.concat([franchise_sheets[y] for y in franchise_sheets if selected_year == "All" or str(y) == str(selected_year)], ignore_index=True)

# # Clean up
# df.columns = df.columns.str.strip()
# for c in ["State", "District", "Franchise Owner Name", "Franchise Name ", "Franchise Name"]:
#     if c in df.columns:
#         df[c] = df[c].astype(str).str.strip()

# # --- Sidebar: More filters ---
# states = sorted(df['State'].dropna().unique()) if 'State' in df.columns else []
# sel_states = st.sidebar.multiselect("Filter by State", states, default=states)
# districts = sorted(df['District'].dropna().unique()) if 'District' in df.columns else []
# sel_districts = st.sidebar.multiselect("Filter by District", districts, default=districts)
# search_sidebar = st.sidebar.text_input("Search Franchisee/Owner (partial match)", key="Franchise/Owner_search")


# # --- Apply filters ---
# filtered_df = df.copy()
# if sel_states:
#     filtered_df = filtered_df[filtered_df['State'].isin(sel_states)]
# if sel_districts:
#     filtered_df = filtered_df[filtered_df['District'].isin(sel_districts)]
# if search_sidebar:
#     mask = filtered_df.apply(lambda row: search_sidebar.lower() in str(row).lower(), axis=1)
#     filtered_df = filtered_df[mask]

# st.subheader(f"Franchise Data ({'All Years' if selected_year == 'All' else selected_year})")
# st.dataframe(filtered_df, use_container_width=True)

# # --- KPIs ---
# col1, col2, col3 = st.columns(3)
# col1.metric("Total Franchisees", filtered_df.shape[0])
# if 'Revenue Collected till Date (in Rs.)' in filtered_df.columns:
#     filtered_df['Revenue Collected till Date (in Rs.)'] = pd.to_numeric(filtered_df['Revenue Collected till Date (in Rs.)'].str.replace(",", ""), errors='coerce')
#     col2.metric("Total Revenue (₹)", f"{filtered_df['Revenue Collected till Date (in Rs.)'].sum():,.0f}")
# if 'State' in filtered_df.columns:
#     col3.metric("States Covered", filtered_df['State'].nunique())

# # --- Year-wise trends ---
# if 'Year' in filtered_df.columns and 'Revenue Collected till Date (in Rs.)' in filtered_df.columns:
#     yearly_rev = filtered_df.groupby('Year')['Revenue Collected till Date (in Rs.)'].sum().reset_index()
#     fig = px.line(yearly_rev, x='Year', y='Revenue Collected till Date (in Rs.)', markers=True, title='Year-wise Revenue Trend')
#     st.plotly_chart(fig, use_container_width=True)

# if 'Year' in filtered_df.columns:
#     yearly_count = filtered_df.groupby('Year').size().reset_index(name="Franchisee Count")
#     fig2 = px.bar(yearly_count, x='Year', y='Franchisee Count', title='Franchisees Added Each Year')
#     st.plotly_chart(fig2, use_container_width=True)

# # --- Franchisees by State ---
# if 'State' in filtered_df.columns:
#     state_counts = filtered_df['State'].value_counts().reset_index()
#     state_counts.columns = ['State', 'Count']
#     fig3 = px.bar(state_counts, x='State', y='Count', title='Franchisees by State')
#     st.plotly_chart(fig3, use_container_width=True)

# # --- Franchisees by District ---
# if 'District' in filtered_df.columns:
#     district_counts = filtered_df['District'].value_counts().reset_index()
#     district_counts.columns = ['District', 'Count']
#     fig4 = px.bar(district_counts, x='District', y='Count', title='Franchisees by District')
#     st.plotly_chart(fig4, use_container_width=True)

# # --- Payment Status Pie ---
# if 'Remarks (if any)' in filtered_df.columns:
#     payment_status = filtered_df['Remarks (if any)'].value_counts().reset_index()
#     payment_status.columns = ['Status', 'Count']
#     fig5 = px.pie(payment_status, names='Status', values='Count', title='Payment Status Distribution')
#     st.plotly_chart(fig5, use_container_width=True)

# # # --- Franchisee/Owner search results ---
# # if search_text:
# #     st.success(f"Found {filtered_df.shape[0]} record(s) for search: {search_text}")

st.info("Dashboard updates automatically as new data is added to the sheets. Use the sidebar filters to drill down by year, category, or state for deeper insights.")
