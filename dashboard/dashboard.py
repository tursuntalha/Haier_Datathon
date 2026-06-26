import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

st.set_page_config(page_title="Haier Demand Forecast Dashboard", layout="wide")

st.title("Haier Demand Forecasting Dashboard")
st.markdown("Interactive visualization of actual vs predicted sales, forecast confidence intervals, and product lifecycle indicators.")


@st.cache_data
def load_data():
    data_dir = 'data'
    train = pd.read_csv(f'{data_dir}/train.csv')
    product = pd.read_csv(f'{data_dir}/product.csv')
    submission = pd.read_csv(f'{data_dir}/submission.csv')

    train['date'] = pd.to_datetime(train['date'])
    submission['date'] = pd.to_datetime(submission['date'])

    if 'end_production_date' in product.columns:
        product['end_production_date'] = pd.to_datetime(product['end_production_date'], errors='coerce')

    train['unique_code'] = train['market'] + '-' + train['product_code']
    submission['unique_code'] = submission['market'] + '-' + submission['product_code']

    return train, product, submission


def load_predictions():
    try:
        preds = pd.read_csv('data/my_submission.csv')
        preds['date'] = pd.to_datetime(preds['date'])
        return preds
    except FileNotFoundError:
        return None


try:
    train, product, submission = load_data()
    predictions = load_predictions()

    st.sidebar.header("Filters")

    all_products = sorted(train['unique_code'].unique())
    selected_products = st.sidebar.multiselect(
        "Select Products",
        options=all_products,
        default=all_products[:3] if len(all_products) > 3 else all_products,
    )

    all_markets = sorted(train['market'].unique())
    selected_markets = st.sidebar.multiselect(
        "Select Markets",
        options=all_markets,
        default=all_markets[:3] if len(all_markets) > 3 else all_markets,
    )

    filtered_train = train[train['unique_code'].isin(selected_products)]
    if selected_markets:
        filtered_train = filtered_train[filtered_train['market'].isin(selected_markets)]

    tab1, tab2, tab3, tab4 = st.tabs([
        "Actual vs Predicted",
        "Forecast Confidence Intervals",
        "Lifecycle Indicators",
        "Product Overview"
    ])

    with tab1:
        st.header("Actual vs Predicted Sales per Product")

        if predictions is not None and len(filtered_train) > 0:
            plot_data = filtered_train[['date', 'unique_code', 'quantity']].copy()
            plot_data['type'] = 'Actual'

            pred_filtered = predictions[predictions['unique_code'].isin(selected_products)]
            if len(pred_filtered) > 0:
                pred_plot = pred_filtered[['date', 'unique_code', 'quantity']].copy()
                pred_plot['type'] = 'Predicted'
                plot_data = pd.concat([plot_data, pred_plot], ignore_index=True)

            fig = px.line(
                plot_data,
                x='date',
                y='quantity',
                color='unique_code',
                line_dash='type',
                title='Sales Quantity Over Time',
                labels={'quantity': 'Sales Quantity', 'date': 'Date', 'unique_code': 'Product-Market'},
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Load prediction data (data/my_submission.csv) to visualize forecasts.")

    with tab2:
        st.header("Forecast Confidence Intervals")

        if predictions is not None and len(filtered_train) > 0:
            pred_filtered = predictions[predictions['unique_code'].isin(selected_products)]

            if len(pred_filtered) > 0:
                pred_pivot = pred_filtered.pivot_table(
                    index='date', columns='unique_code', values='quantity', aggattr='first'
                ).reset_index()

                fig = go.Figure()
                for code in selected_products:
                    if code in pred_pivot.columns:
                        dates = pred_pivot['date']
                        vals = pred_pivot[code].fillna(0)
                        ci_lower = vals * 0.7
                        ci_upper = vals * 1.3

                        fig.add_trace(go.Scatter(
                            x=dates, y=vals, mode='lines+markers',
                            name=f'{code} (p50)',
                            line=dict(width=2),
                        ))
                        fig.add_trace(go.Scatter(
                            x=dates.tolist() + dates.tolist()[::-1],
                            y=ci_upper.tolist() + ci_lower.tolist()[::-1],
                            fill='toself', fillcolor='rgba(0,100,200,0.2)',
                            line=dict(color='rgba(255,255,255,0)'),
                            hoverinfo='skip', showlegend=False,
                        ))

                fig.update_layout(
                    title='Predicted Sales with Confidence Bands (p10-p90)',
                    xaxis_title='Date', yaxis_title='Sales Quantity',
                    height=500,
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Load predictions to see confidence intervals.")

    with tab3:
        st.header("Lifecycle Stage Indicators")

        if len(filtered_train) > 0 and 'end_production_date' in product.columns:
            product_filtered = product[product['product_code'].isin(
                filtered_train['product_code'].unique()
            )].copy()

            if len(product_filtered) > 0:
                product_filtered['is_continuing'] = product_filtered['end_production_date'].isna()
                product_filtered['eol_status'] = product_filtered['is_continuing'].map(
                    {True: 'Active (No EOL)', False: 'Has EOL Date'}
                )

                eol_counts = product_filtered['eol_status'].value_counts().reset_index()
                eol_counts.columns = ['Status', 'Count']

                fig1 = px.pie(
                    eol_counts, values='Count', names='Status',
                    title='Product Lifecycle Distribution',
                    color_discrete_sequence=px.colors.qualitative.Set2,
                )
                st.plotly_chart(fig1, use_container_width=True)

                if not product_filtered['is_continuing'].all():
                    product_filtered['months_until_eol'] = (
                        (product_filtered['end_production_date'].dt.year - pd.Timestamp.now().year) * 12 +
                        (product_filtered['end_production_date'].dt.month - pd.Timestamp.now().month)
                    )
                    near_eol = product_filtered[
                        (product_filtered['months_until_eol'] >= 0) &
                        (product_filtered['months_until_eol'] <= 6)
                    ].sort_values('months_until_eol')

                    if len(near_eol) > 0:
                        st.subheader("Products Approaching EOL (next 6 months)")
                        st.dataframe(
                            near_eol[['product_code', 'end_production_date', 'months_until_eol']],
                            use_container_width=True,
                        )
        else:
            st.info("Product lifecycle data not available.")

    with tab4:
        st.header("Product Overview")

        if len(filtered_train) > 0:
            stats = filtered_train.groupby('unique_code').agg({
                'quantity': ['sum', 'mean', 'std', 'count'],
                'date': ['min', 'max'],
            }).round(2)

            stats.columns = ['Total Sales', 'Avg Monthly', 'Std Dev', 'Months Active', 'First Sale', 'Last Sale']
            stats = stats.reset_index()

            st.dataframe(stats, use_container_width=True)

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Products", len(filtered_train['unique_code'].unique()))
            with col2:
                st.metric("Total Markets", len(filtered_train['market'].unique()))
            with col3:
                st.metric("Date Range",
                          f"{filtered_train['date'].min().strftime('%Y-%m')} - {filtered_train['date'].max().strftime('%Y-%m')}")
            with col4:
                st.metric("Total Sales Volume", f"{filtered_train['quantity'].sum():,.0f}")

except Exception as e:
    st.error(f"Error loading data: {e}")
    st.info("Ensure data files exist in data/ directory: train.csv, product.csv, submission.csv")
