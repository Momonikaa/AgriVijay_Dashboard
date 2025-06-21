import streamlit as st
import pandas as pd
import plotly.express as px

# Load data and clean headers
@st.cache_data
def load_data():
    detailed = pd.read_csv('data/detailed_transactions.csv')
    sales = pd.read_csv('data/product_sales_metrics.csv')
    detailed.columns = detailed.columns.str.strip()
    sales.columns = sales.columns.str.strip()
    return detailed, sales

detailed, sales = load_data()

st.title("Farmer Data Dashboard")

# --- Product Sales Metrics Sheet Visualizations ---
st.header("Product Sales Metrics Overview")

if 'Product Name' in sales.columns and 'No of Units' in sales.columns:
    fig_units = px.bar(
        sales,
        x='Product Name',
        y='No of Units',
        color='Category',
        title='Units Sold per Product'
    )
    st.plotly_chart(fig_units)

if 'Product Name' in sales.columns and 'Revenue Contribution in INR' in sales.columns:
    fig_revenue = px.bar(
        sales,
        x='Product Name',
        y='Revenue Contribution in INR',
        color='Category',
        title='Revenue per Product'
    )
    st.plotly_chart(fig_revenue)

# --- Detailed Transactions Sheet Visualizations ---
st.header("Detailed Transactions Analysis")

# Sales by State
if 'State/Branch' in detailed.columns and 'Product' in detailed.columns:
    state_sales = detailed.groupby('State/Branch').size().reset_index(name='Count')
    fig_state = px.bar(
        state_sales,
        x='State/Branch',
        y='Count',
        title='Number of Transactions by State/Branch'
    )
    st.plotly_chart(fig_state)

# Sales by Product Category
if 'Product Category' in detailed.columns:
    prod_cat_sales = detailed['Product Category'].value_counts().reset_index()
    prod_cat_sales.columns = ['Product Category', 'Count']
    fig_cat = px.pie(
        prod_cat_sales,
        names='Product Category',
        values='Count',
        title='Sales by Product Category'
    )
    st.plotly_chart(fig_cat)

# Sales over Time
if 'Date' in detailed.columns:
    detailed['Date'] = pd.to_datetime(detailed['Date'], errors='coerce')
    sales_over_time = detailed.groupby(detailed['Date'].dt.to_period('M')).size().reset_index(name='Count')
    sales_over_time['Date'] = sales_over_time['Date'].astype(str)
    fig_time = px.line(
        sales_over_time,
        x='Date',
        y='Count',
        title='Sales Trend Over Time'
    )
    st.plotly_chart(fig_time)

# Unique Customers
if 'Customer' in detailed.columns:
    unique_customers = detailed['Customer'].nunique()
    st.metric("Unique Customers", unique_customers)

# Top Products Table
if 'Product' in detailed.columns:
    st.subheader("Top Products")
    top_products = detailed['Product'].value_counts().reset_index()
    top_products.columns = ['Product', 'Count']
    st.dataframe(top_products.head(10))

# GHG/CO2 Emissions Abated
if 'GHG / CO2 Emissions Abated / Year' in detailed.columns:
    st.subheader("Total GHG / CO2 Emissions Abated (Yearly)")
    total_ghg = detailed['GHG / CO2 Emissions Abated / Year'].sum()
    st.write(f"{total_ghg}")

# --- Additional Analysis ---
st.header("Additional Insights")

# 1. Average Invoice Amount per Product Category
if 'Product Category' in detailed.columns and 'Invoice Amount(in Rs.)' in detailed.columns:
    avg_invoice = detailed.groupby('Product Category')['Invoice Amount(in Rs.)'].mean().reset_index()
    st.subheader("Average Invoice Amount per Product Category")
    st.dataframe(avg_invoice)

# 2. Repeat Customers Count
if 'Customer' in detailed.columns:
    repeat_customers = detailed.groupby('Customer').size().reset_index(name='purchase_count')
    repeat_customers_count = repeat_customers[repeat_customers['purchase_count'] > 1].shape[0]
    st.subheader("Repeat Customers (Purchased More Than Once)")
    st.write(f"Number of repeat customers: {repeat_customers_count}")

# 3. Top 5 Customers by Total Invoice Amount
if 'Customer' in detailed.columns and 'Invoice Amount(in Rs.)' in detailed.columns:
    top_customers = detailed.groupby('Customer')['Invoice Amount(in Rs.)'].sum().reset_index().sort_values(by='Invoice Amount(in Rs.)', ascending=False).head(5)
    st.subheader("Top 5 Customers by Total Invoice Amount")
    st.dataframe(top_customers)

# 4. Monthly Revenue Trend
if 'Date' in detailed.columns and 'Invoice Amount(in Rs.)' in detailed.columns:
    detailed['Date'] = pd.to_datetime(detailed['Date'], errors='coerce')
    monthly_revenue = detailed.groupby(detailed['Date'].dt.to_period('M'))['Invoice Amount(in Rs.)'].sum().reset_index()
    monthly_revenue['Date'] = monthly_revenue['Date'].astype(str)
    fig_monthly_revenue = px.line(
        monthly_revenue,
        x='Date',
        y='Invoice Amount(in Rs.)',
        title='Monthly Revenue Trend'
    )
    st.plotly_chart(fig_monthly_revenue)

# 5. Product Category Contribution to Total Revenue
if 'Product Category' in detailed.columns and 'Invoice Amount(in Rs.)' in detailed.columns:
    revenue_by_category = detailed.groupby('Product Category')['Invoice Amount(in Rs.)'].sum().reset_index()
    st.subheader("Revenue by Product Category")
    st.dataframe(revenue_by_category)
    fig_revenue_cat = px.pie(
        revenue_by_category,
        names='Product Category',
        values='Invoice Amount(in Rs.)',
        title='Revenue Share by Product Category'
    )
    st.plotly_chart(fig_revenue_cat)

# 6. Average GHG/CO2 Emissions Abated per Product Category
if 'Product Category' in detailed.columns and 'GHG / CO2 Emissions Abated / Year' in detailed.columns:
    avg_ghg = detailed.groupby('Product Category')['GHG / CO2 Emissions Abated / Year'].mean().reset_index()
    st.subheader("Average GHG/CO2 Emissions Abated per Product Category")
    st.dataframe(avg_ghg)
    fig_ghg = px.bar(
        avg_ghg,
        x='Product Category',
        y='GHG / CO2 Emissions Abated / Year',
        title='Average GHG/CO2 Emissions Abated per Product Category'
    )
    st.plotly_chart(fig_ghg)

st.write("Dashboard updates automatically as new data is added to the sheets.")
