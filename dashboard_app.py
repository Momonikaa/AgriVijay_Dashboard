import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Farmer Data Dashboard", layout="wide")

# --- Load data and clean headers ---
@st.cache_data
def load_data():
    detailed = pd.read_csv('data/detailed_transactions.csv')
    sales = pd.read_csv('data/product_sales_metrics.csv', header=2)
    detailed.columns = detailed.columns.str.strip()
    sales.columns = sales.columns.str.strip()  # This fixes the column name issue!
    return detailed, sales


detailed, sales = load_data()

# --- Fix column types for correct calculations ---
if 'No of Units' in sales.columns:
    sales['No of Units'] = pd.to_numeric(sales['No of Units'], errors='coerce')
if 'Revenue Contribution in INR' in sales.columns:
    sales['Revenue Contribution in INR'] = pd.to_numeric(sales['Revenue Contribution in INR'], errors='coerce')

if 'Invoice Amount(in Rs.)' in detailed.columns:
    detailed['Invoice Amount(in Rs.)'] = pd.to_numeric(detailed['Invoice Amount(in Rs.)'], errors='coerce')
if 'GHG / CO2 Emissions Abated / Year' in detailed.columns:
    detailed['GHG / CO2 Emissions Abated / Year'] = pd.to_numeric(detailed['GHG / CO2 Emissions Abated / Year'], errors='coerce')

# --- Sidebar Filters ---
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

# --- KPIs ---
st.title("Farmer Data Dashboard")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Revenue (₹)", f"{filtered['Invoice Amount(in Rs.)'].sum():,.0f}")
col2.metric("UnitsSold", f"{sales['No of Units'].sum():.0f}")
col3.metric("GHG Abated (Yearly, tCO2e)", f"{filtered['GHG / CO2 Emissions Abated / Year'].sum():,.2f}")
col4.metric("Unique Customers", f"{filtered['Customer Name'].nunique()}")

# --- Product Sales Metrics Overview ---
st.header("Product Sales Metrics Overview")

if 'Product Name' in sales.columns and 'No of Units' in sales.columns:
    fig_Units= px.bar(
        sales,
        x='Product Name',
        y='No of Units',
        color='Category',
        title='UnitsSold per Product'
    )
    st.plotly_chart(fig_Units, use_container_width=True)

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

st.header("🔎 Customer Business Records Lookup")

# Get unique customer names (drop NaN and sort for usability)
customer_names = detailed['Customer Name'].dropna().unique()
customer_names = sorted(customer_names)

# Option 1: Dropdown selection
selected_customer = st.selectbox("Select a customer to view all their records:", customer_names)

# Option 2: Text search (for partial matches, case-insensitive)
search_text = st.text_input("Or search by customer name (partial match):", "")

# Filter logic
if search_text:
    # Case-insensitive partial match
    customer_records = detailed[detailed['Customer Name'].str.contains(search_text, case=False, na=False)]
    st.write(f"Showing results for search: **{search_text}**")
elif selected_customer:
    customer_records = detailed[detailed['Customer Name'] == selected_customer]
    st.write(f"Showing all records for: **{selected_customer}**")
else:
    customer_records = pd.DataFrame()  # Empty by default

# Display results
if not customer_records.empty:
    st.dataframe(customer_records)
    st.success(f"Found {len(customer_records)} record(s) for this customer.")
else:
    st.info("No records found for the given customer.")


st.info("Dashboard updates automatically as new data is added to the sheets. Use the sidebar filters to drill down by year, category, or state for deeper insights.")
