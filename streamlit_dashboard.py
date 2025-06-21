import streamlit as st
import pandas as pd
import plotly.express as px
from textblob import TextBlob
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

st.set_page_config(page_title="📊 Sentiment & Stock Dashboard", layout="wide", initial_sidebar_state="expanded")

# --- Theme Toggle and Style ---
st.sidebar.header("Settings")
selected_theme = st.sidebar.radio("Choose Theme", ["Light", "Dark"])
theme_bg = "#1e1e1e" if selected_theme == "Dark" else "#f7f4e9"
theme_fg = "#ffffff" if selected_theme == "Dark" else "#333333"
theme_style = f"""
    <style>
    body {{ background-color: {theme_bg}; color: {theme_fg}; }}
    .reportview-container .main .block-container{{ color: {theme_fg}; }}
    .stButton > button {{
        font-weight: bold;
        background-color: #ffcf33;
        color: {theme_fg};
        border-radius: 10px;
        padding: 10px 20px;
        margin: 5px;
    }}
    </style>
"""
st.markdown(theme_style, unsafe_allow_html=True)

st.title("📈 Social Media Sentiment + Stock Price Dashboard")
st.caption("Analyze, Predict, and Strategize with Sentiment and Market Data")

# --- Sidebar Inputs ---
st.sidebar.header("Upload Datasets")
social_file = st.sidebar.file_uploader("Upload Social Media CSV", type="csv")
stock_file = st.sidebar.file_uploader("Upload Stock Market CSV", type="csv")

# --- Load Data ---
social_df, stock_df = None, None
if social_file:
    social_df = pd.read_csv(social_file)
    st.success(f"Social data loaded: {len(social_df)} records")
if stock_file:
    stock_df = pd.read_csv(stock_file)
    st.success(f"Stock data loaded: {len(stock_df)} records")

# --- Action Buttons ---
cols = st.columns(4)
run_diag = cols[0].button("📊 Diagnostic Analysis")
run_pred = cols[1].button("🔮 Predict Next-Day Price")
run_presc = cols[2].button("📈 Prescriptive Analysis")
run_perf = cols[3].button("📉 Performance Analysis")

# --- Diagnostic Analysis ---
if run_diag and social_df is not None and stock_df is not None:
    st.markdown("---")
    st.subheader("📊 Diagnostic Analysis")
    try:
        social_df['date'] = pd.to_datetime(social_df['date'], errors='coerce')
        social_df.dropna(subset=['date', 'text'], inplace=True)
        social_df['sentiment_score'] = social_df['text'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
        social_df['sentiment_category'] = social_df['sentiment_score'].apply(lambda x: 'positive' if x > 0 else ('negative' if x < 0 else 'neutral'))

        sentiment_counts = social_df['sentiment_category'].value_counts()
        fig_pie = px.pie(
            names=sentiment_counts.index,
            values=sentiment_counts.values,
            title="Sentiment Distribution",
            color_discrete_sequence=["#22eeaa", "#8888aa", "#ff5566"]
        )
        st.plotly_chart(fig_pie)

        sentiment_trend = social_df.groupby(social_df['date'].dt.date)['sentiment_score'].mean().reset_index()
        stock_df['date'] = pd.to_datetime(stock_df['date'], errors='coerce')
        stock_df['date'] = stock_df['date'].dt.date
        merged = pd.merge(sentiment_trend, stock_df[['date', 'close']], on='date', how='inner')

        fig_corr = px.line(merged, x='date', y=['sentiment_score', 'close'],
                          title="Sentiment vs Stock Close Price")
        st.plotly_chart(fig_corr)

    except Exception as e:
        st.error(f"Error in Diagnostic Analysis: {e}")

# --- Predict Next-Day Price ---
if run_pred and social_df is not None and stock_df is not None:
    st.markdown("---")
    st.subheader("🔮 Predict Next-Day Price")
    try:
        social_df['date'] = pd.to_datetime(social_df['date'], errors='coerce')
        social_df['sentiment_score'] = social_df['text'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
        sentiment_trend = social_df.groupby(social_df['date'].dt.date)['sentiment_score'].mean().reset_index()

        stock_df['date'] = pd.to_datetime(stock_df['date'], errors='coerce')
        stock_df['date'] = stock_df['date'].dt.date
        data = pd.merge(stock_df[['date', 'close']], sentiment_trend, on='date', how='inner')
        data = data.sort_values('date')
        data['next_close'] = data['close'].shift(-1)
        data.dropna(inplace=True)
        data['price_diff'] = data['next_close'] - data['close']

        X = data[['close', 'sentiment_score']]
        y = data['price_diff']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)

        st.write(f"📉 Mean Squared Error: {mse:.4f}")

        df_plot = pd.DataFrame({
            'Index': range(len(y_test)),
            'Actual': y_test.values,
            'Predicted': y_pred
        })
        fig_pred = px.line(df_plot, x='Index', y=['Actual', 'Predicted'], title="Actual vs Predicted Price Change")
        st.plotly_chart(fig_pred)

    except Exception as e:
        st.error(f"Error in Prediction: {e}")

# --- Prescriptive Analysis ---
if run_presc and social_df is not None and stock_df is not None:
    st.markdown("---")
    st.subheader("📈 Prescriptive Analysis")
    try:
        social_df['date'] = pd.to_datetime(social_df['date'], errors='coerce')
        social_df['sentiment_score'] = social_df['text'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
        sentiment_trend = social_df.groupby(social_df['date'].dt.date)['sentiment_score'].mean().reset_index()
        stock_df['date'] = pd.to_datetime(stock_df['date'], errors='coerce')
        stock_df['date'] = stock_df['date'].dt.date
        merged = pd.merge(sentiment_trend, stock_df[['date', 'close']], on='date', how='inner')
        merged = merged.sort_values('date')
        merged['price_diff'] = merged['close'].diff()
        merged['buy_signal'] = (merged['sentiment_score'] > 0.5) & (merged['price_diff'] < 0)

        buy_dates = merged[merged['buy_signal']]['date'].tolist()
        st.write(f"📌 Buy signals found: {len(buy_dates)}")
        if buy_dates:
            st.write("Top Buy Signal Dates:", buy_dates[:10])

        negative_sent = merged[merged['sentiment_score'] < 0]
        if not negative_sent.empty:
            corr_neg = negative_sent['sentiment_score'].corr(negative_sent['price_diff'])
            if corr_neg < -0.3:
                st.info("Strong negative sentiment correlates with price drops. Avoid product releases during such periods.")
            else:
                st.info("No strong negative sentiment correlation with price. Sentiment-based timing less critical.")
        else:
            st.warning("Not enough negative sentiment data for analysis.")

    except Exception as e:
        st.error(f"Error in Prescriptive Analysis: {e}")

# --- Performance Analysis ---
if run_perf and social_df is not None and stock_df is not None:
    st.markdown("---")
    st.subheader("📉 Performance Analysis")
    try:
        social_df['date'] = pd.to_datetime(social_df['date'], errors='coerce')
        stock_df['date'] = pd.to_datetime(stock_df['date'], errors='coerce')
        stock_df = stock_df.sort_values('date')
        social_df['sentiment_score'] = social_df['text'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
        sentiment_trend = social_df.groupby(social_df['date'].dt.date)['sentiment_score'].mean().reset_index()

        stock_df['date'] = stock_df['date'].dt.date
        merged = pd.merge(stock_df[['date', 'close']], sentiment_trend, on='date', how='inner')
        merged = merged.sort_values('date')

        merged['signal'] = merged['sentiment_score'].apply(lambda x: 1 if x > 0.5 else 0)
        merged['returns'] = merged['close'].pct_change()
        merged['strategy_returns'] = merged['signal'].shift(1) * merged['returns']

        merged.dropna(inplace=True)

        merged['cumulative_market'] = (1 + merged['returns']).cumprod()
        merged['cumulative_strategy'] = (1 + merged['strategy_returns']).cumprod()

        fig_perf = px.line(merged, x='date', y=['cumulative_market', 'cumulative_strategy'],
                          title="Cumulative Returns: Market vs Strategy",
                          labels={'value': 'Cumulative Return', 'date': 'Date', 'variable': 'Legend'})
        st.plotly_chart(fig_perf, use_container_width=True)

        market_return = merged['cumulative_market'].iloc[-1] - 1
        strategy_return = merged['cumulative_strategy'].iloc[-1] - 1
        st.write(f"📈 Final Strategy Return: {strategy_return:.2%}")
        st.write(f"📉 Final Market Return: {market_return:.2%}")

    except Exception as e:
        st.error(f"Error in Performance Analysis: {e}")

# --- Footer ---
st.markdown("""
<hr style="margin-top:3rem;">
<p style="text-align: center; font-size: 0.9rem;">Made with ❤️ using Streamlit</p>
""", unsafe_allow_html=True)
