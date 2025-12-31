import streamlit as st
import akshare as ak
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import concurrent.futures
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from openai import OpenAI
import requests
import json
import os

# === 0. 全局配置 & 密钥管理系统 ===
CONFIG_FILE = ".mp_config.json"

def load_saved_key():
    """从本地文件加载 API Key"""
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                config = json.load(f)
                return config.get("api_key", "")
        except:
            return ""
    return ""

def save_key_locally(key):
    """保存 API Key 到本地文件"""
    try:
        with open(CONFIG_FILE, "w") as f:
            json.dump({"api_key": key}, f)
        return True
    except:
        return False

# === 1. 页面配置 ===
st.set_page_config(
    page_title="MarketPulse Pro",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# === 2. UI 样式 ===
st.markdown("""
<style>
    .stApp { background-color: #f8f9fa; font-family: -apple-system, sans-serif; }
    .stock-row { background-color: white; padding: 15px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); border: 1px solid #e9ecef; margin-bottom: 10px; }
    .badge-up { background-color: #ffeaea; color: #d9001b; padding: 4px 8px; border-radius: 4px; font-weight: bold; }
    .badge-down { background-color: #eafbf2; color: #00a854; padding: 4px 8px; border-radius: 4px; font-weight: bold; }
    .stButton button { border-radius: 6px; height: 2.5rem; }
    div[data-testid="stDialog"] { background-color: #ffffff; }
    .ai-box { background-color: #eff6ff; border-left: 4px solid #3b82f6; padding: 15px; border-radius: 6px; font-size: 0.95rem; line-height: 1.6; color: #1e293b; margin-top: 10px; }
    .key-status { background-color: #dcfce7; color: #166534; padding: 8px 12px; border-radius: 6px; border: 1px solid #bbf7d0; font-size: 0.9rem; margin-bottom: 10px; }
    
    /* 指标小标签 */
    .tag-trend-bull { color: #d9001b; font-weight: bold; font-size: 0.9em; }
    .tag-trend-bear { color: #00a854; font-weight: bold; font-size: 0.9em; }
    .tag-signal-gold { background: #fff7e6; color: #d46b08; padding: 2px 6px; border-radius: 4px; font-size: 0.85em; border: 1px solid #ffd591; }
    .tag-signal-death { background: #f6ffed; color: #389e0d; padding: 2px 6px; border-radius: 4px; font-size: 0.85em; border: 1px solid #b7eb8f; }
</style>
""", unsafe_allow_html=True)

# === 3. 数据层 (Data Layer) ===

@st.cache_data(ttl=3600)
def get_all_stock_names_map():
    try:
        df = ak.stock_zh_a_spot_em()
        return dict(zip(df['代码'], df['名称']))
    except: return {}

@st.cache_data(ttl=3600)
def get_dynamic_pool(sector_name, limit=5):
    try:
        df = ak.stock_board_industry_cons_em(symbol=sector_name)
    except:
        try: df = ak.stock_board_concept_cons_em(symbol=sector_name)
        except: return {}
    try:
        if '总市值' in df.columns:
            df = df.sort_values(by='总市值', ascending=False)
        top_stocks = df.head(limit)
        return dict(zip(top_stocks['代码'], top_stocks['名称']))
    except: return {}

@st.cache_data(ttl=86400)
def get_all_sectors():
    try:
        df = ak.stock_board_industry_name_em()
        lst = df['板块名称'].tolist() + ["低空经济", "人工智能", "算力概念", "中特估", "华为概念", "新能源车", "固态电池", "量子科技", "人形机器人"]
        return sorted(list(set(lst)))
    except: return ["半导体", "银行", "证券"]

@st.cache_data(ttl=15)
def fetch_stock_min_data(code, period='5'):
    """获取分钟K线 (带缓存，极速列表用)"""
    try:
        return ak.stock_zh_a_hist_min_em(symbol=code, period=period, adjust="qfq")
    except:
        return pd.DataFrame()

def get_kline_data_uncached(code, period='daily'):
    """获取弹窗用的详细K线 (不缓存，保证最新)"""
    try:
        if period in ['daily', 'weekly', 'monthly']:
            start_date = (datetime.now() - timedelta(days=365*2)).strftime("%Y%m%d")
            return ak.stock_zh_a_hist(symbol=code, period=period, start_date=start_date, adjust="qfq")
        else:
            return ak.stock_zh_a_hist_min_em(symbol=code, period=period, adjust="qfq")
    except:
        return pd.DataFrame()

# === 4. 逻辑处理层 ===

def calculate_tech_indicators(df):
    """通用指标计算函数"""
    if df.empty or len(df) < 30: return None
    df['MA5'] = df['收盘'].rolling(5).mean()
    df['MA20'] = df['收盘'].rolling(20).mean()
    exp12 = df['收盘'].ewm(span=12, adjust=False).mean()
    exp26 = df['收盘'].ewm(span=26, adjust=False).mean()
    df['DIF'] = exp12 - exp26
    df['DEA'] = df['DIF'].ewm(span=9, adjust=False).mean()
    df['MACD'] = (df['DIF'] - df['DEA']) * 2
    delta = df['收盘'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + (gain / loss)))
    low_9 = df['最低'].rolling(9).min(); high_9 = df['最高'].rolling(9).max()
    rsv = (df['收盘'] - low_9) / (high_9 - low_9) * 100
    df['K'] = rsv.ewm(com=2).mean(); df['D'] = df['K'].ewm(com=2).mean(); df['J'] = 3 * df['K'] - 2 * df['D']
    latest = df.iloc[-1]
    return {
        "price": latest['收盘'], "ma5": latest['MA5'], "ma20": latest['MA20'],
        "rsi": latest['RSI'], "macd": latest['MACD'], "dif": latest['DIF'], "dea": latest['DEA'],
        "k": latest['K'], "d": latest['D'], "j": latest['J'],
        "trend": "多头" if latest['收盘'] > latest['MA20'] else "空头"
    }

def process_single_stock_fast(code, name):
    """
    [极速模式] 单只股票处理逻辑 - 列表专用
    包含：现价、涨跌、趋势、MACD、KDJ
    """
    try:
        # 获取 5 分钟线
        df = fetch_stock_min_data(code, '5')
        
        if df.empty:
            return {
                "代码": code, "名称": name, "最新价": 0.0, "今日涨跌": 0.0,
                "趋势": "⏳ 离线", "成交量": 0, "近期走势": [],
                "MACD信号": "-", "KDJ信号": "-"
            }
        
        # 1. 基础行情
        latest = df.iloc[-1]
        price = latest['收盘']
        today_str = datetime.now().strftime("%Y-%m-%d")
        df_today = df[df['时间'].str.contains(today_str)]
        
        if not df_today.empty:
            open_price = df_today.iloc[0]['开盘']
            pct = (price - open_price) / open_price
        else:
            pct = (price - df.iloc[0]['开盘']) / df.iloc[0]['开盘']

        # 2. 计算核心指标 (复用逻辑)
        tech = calculate_tech_indicators(df)
        
        trend_str = "⚪"
        macd_str = "-"
        kdj_str = "-"
        
        if tech:
            trend_str = "🔴多头" if tech['trend'] == "多头" else "💚空头"
            macd_str = "📈金叉" if tech['dif'] > tech['dea'] else "📉死叉"
            if tech['j'] < 0: kdj_str = "💎超卖"
            elif tech['j'] > 100: kdj_str = "⚠️超买"
            else: kdj_str = f"J:{int(tech['j'])}"

        return {
            "代码": code, 
            "名称": name, 
            "最新价": price, 
            "今日涨跌": pct, 
            "趋势": trend_str,
            "MACD信号": macd_str,
            "KDJ信号": kdj_str,
            "成交量": latest['成交量'],
            "近期走势": df['收盘'].tail(50).tolist()
        }
    except Exception:
        return None

def call_llm_stream(api_key, api_base, model, prompt):
    client = OpenAI(api_key=api_key, base_url=api_base)
    stream = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}], stream=True
    )
    return stream

# === 5. 可视化组件 ===

def plot_trend_chart(df, title, height=500):
    """绘制平滑分时走势图"""
    if df.empty:
        st.warning("暂无数据")
        return
    x_axis = df['时间'] if '时间' in df.columns else df.index
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.03)
    
    # 现价线
    fig.add_trace(go.Scatter(x=x_axis, y=df['收盘'], mode='lines', line=dict(color='#2962ff', width=2), fill='tozeroy', fillcolor='rgba(41, 98, 255, 0.1)', name='现价'), row=1, col=1)
    # 均价线
    avg_price = df['收盘'].expanding().mean()
    fig.add_trace(go.Scatter(x=x_axis, y=avg_price, mode='lines', line=dict(color='#ff9900', width=1, dash='dash'), name='均价'), row=1, col=1)
    # 成交量
    colors = ['#d9001b' if c >= o else '#00a854' for c, o in zip(df['收盘'], df['开盘'])]
    fig.add_trace(go.Bar(x=x_axis, y=df['成交量'], marker_color=colors, name='成交量'), row=2, col=1)
    
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), xaxis_rangeslider_visible=False, showlegend=False, plot_bgcolor='white', paper_bgcolor='white', hovermode='x unified')
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#f0f0f0', row=1, col=1)
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#f0f0f0', row=1, col=1)
    fig.update_yaxes(showgrid=False, row=2, col=1)
    st.plotly_chart(fig, use_container_width=True)

def plot_kline_chart(df, title, height=500):
    """绘制K线图"""
    if df.empty:
        st.warning("暂无数据")
        return
    x_axis = df['日期'] if '日期' in df.columns else df.index
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.03)
    
    fig.add_trace(go.Candlestick(x=x_axis, open=df['开盘'], high=df['最高'], low=df['最低'], close=df['收盘'], increasing_line_color='#d9001b', decreasing_line_color='#00a854', name='价格'), row=1, col=1)
    ma5 = df['收盘'].rolling(5).mean(); ma20 = df['收盘'].rolling(20).mean()
    fig.add_trace(go.Scatter(x=x_axis, y=ma5, line=dict(color='#ff9900', width=1), name='MA5'), row=1, col=1)
    fig.add_trace(go.Scatter(x=x_axis, y=ma20, line=dict(color='#2962ff', width=1), name='MA20'), row=1, col=1)
    
    colors = ['#d9001b' if c >= o else '#00a854' for c, o in zip(df['收盘'], df['开盘'])]
    fig.add_trace(go.Bar(x=x_axis, y=df['成交量'], marker_color=colors, name='成交量'), row=2, col=1)
    
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), xaxis_rangeslider_visible=False, showlegend=False, plot_bgcolor='white', paper_bgcolor='white', hovermode='x unified')
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#f0f0f0', row=1, col=1)
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#f0f0f0', row=1, col=1)
    fig.update_yaxes(showgrid=False, row=2, col=1)
    st.plotly_chart(fig, use_container_width=True)

# === 6. 弹窗逻辑 ===

@st.dialog("📈 个股全景分析终端", width="large")
def open_stock_modal(code, name):
    st.markdown(f"### {name} ({code})")
    tab1, tab2, tab3, tab4, tab5, tab_ai = st.tabs(["⏱️ 分时", "📅 5日", "📈 日K", "🗓️ 周K", "📅 月K", "🤖 AI 参谋"])
    
    with tab1: 
        with st.spinner("加载实时..."): plot_trend_chart(get_kline_data_uncached(code, '1').tail(240), "分时", 400)
    with tab2:
        with st.spinner("加载5日..."): plot_trend_chart(get_kline_data_uncached(code, '5').tail(240), "5日", 400)
    with tab3:
        with st.spinner("加载日线..."): plot_kline_chart(get_kline_data_uncached(code, 'daily').tail(120), "日K", 400)
    with tab4:
        with st.spinner("加载周线..."): plot_kline_chart(get_kline_data_uncached(code, 'weekly').tail(100), "周K", 400)
    with tab5:
        with st.spinner("加载月线..."): plot_kline_chart(get_kline_data_uncached(code, 'monthly').tail(60), "月K", 400)

    with tab_ai:
        st.info("💡 点击下方按钮，调用 AI 大模型结合实时指标进行战术分析")
        saved_key = load_saved_key()
        session_config = st.session_state.get('ai_config', {})
        final_key = session_config.get('api_key') or saved_key
        use_llm = session_config.get('use_llm', False) or (True if final_key else False)
        
        if use_llm and final_key:
            if st.button("🚀 生成深度研报", type="primary", key=f"ai_btn_{code}"):
                with st.spinner("计算指标中..."):
                    df_day = get_kline_data_uncached(code, 'daily')
                    tech = calculate_tech_indicators(df_day)
                    if tech:
                        summary = f"股票：{name} ({code}) | 现价：{tech['price']} | 趋势：{tech['trend']} | MACD：{tech['macd']:.3f} | RSI：{tech['rsi']:.1f} | KDJ_J：{tech['j']:.1f}"
                        prompt = f"你是一名资深A股交易员。请分析：{summary}。给出简短犀利的：1.趋势定性 2.主力意图 3.操作建议。重点加粗。"
                        box = st.empty(); full_resp = ""
                        try:
                            api_base = session_config.get('api_base', "https://dashscope.aliyuncs.com/compatible-mode/v1")
                            api_model = session_config.get('api_model', "qwen-plus")
                            stream = call_llm_stream(final_key, api_base, api_model, prompt)
                            for chunk in stream:
                                if chunk.choices[0].delta.content:
                                    full_resp += chunk.choices[0].delta.content
                                    box.markdown(f"<div class='ai-box'>{full_resp}</div>", unsafe_allow_html=True)
                        except Exception as e: st.error(f"Error: {e}")
                    else: st.error("数据不足")
        else: st.warning("请先在左侧配置 API Key 并保存")

# === 7. 主程序逻辑 ===

if 'my_watchlist' not in st.session_state:
    st.session_state.my_watchlist = ["002236", "300455", "603516", "600895", "603613", "159915", "002415"]
if 'stock_name_cache' not in st.session_state:
    st.session_state.stock_name_cache = {
        "002236": "大华股份", "300455": "航天智装", "603516": "淳中科技", "600895": "张江高科", "603613": "国联股份", "159915": "创业板ETF", "002415": "海康威视"
    }

with st.sidebar:
    st.title("🎛️ 控制台")
    mode = st.radio("模式", ["🔥 动态热点", "⭐ 我的自选"])
    
    with st.expander("🧠 AI 配置 (保险箱)", expanded=True):
        local_key = load_saved_key()
        if local_key:
            st.markdown(f"""<div class="key-status">✅ <b>已激活本地密钥</b><br><span style="font-size:0.8em; opacity:0.8">已隐藏部分内容</span></div>""", unsafe_allow_html=True)
            use_existing_key = True
        else:
            st.info("⚠️ 未检测到本地 Key")
            use_existing_key = False

        enable_edit = st.checkbox("修改/录入 Key", value=False)
        new_key = ""
        if enable_edit or not use_existing_key:
            new_key = st.text_input("API Key", type="password")
            if st.button("💾 保存"):
                if new_key and save_key_locally(new_key):
                    st.success("保存成功！"); time.sleep(1); st.rerun()
        
        current_key = new_key if new_key else local_key
        use_llm = st.checkbox("启用云端 AI", value=True if current_key else False)
        api_base = st.text_input("API Base", value="https://dashscope.aliyuncs.com/compatible-mode/v1")
        api_model = st.text_input("Model", value="qwen-plus")
        st.session_state.ai_config = {"use_llm": use_llm, "api_key": current_key, "api_base": api_base, "api_model": api_model}
    
    st.divider()
    if mode == "⭐ 我的自选":
        with st.expander("🔎 添加股票"):
            with st.spinner("索引加载中..."):
                all_map = get_all_stock_names_map()
            opts = [f"{c} - {all_map.get(c,c)}" for c in all_map] if all_map else []
            def add_cb():
                for i in st.session_state.adder:
                    c = i.split(" - ")[0]; n = i.split(" - ")[1]
                    if c not in st.session_state.my_watchlist:
                        st.session_state.my_watchlist.append(c); st.session_state.stock_name_cache[c] = n
                st.session_state.adder = []
            st.multiselect("搜索", opts, key="adder", on_change=add_cb)
        
        for c in st.session_state.my_watchlist.copy():
            c1, c2 = st.columns([0.8, 0.2])
            c1.caption(f"{st.session_state.stock_name_cache.get(c,c)}")
            if c2.button("✕", key=f"d_{c}"):
                st.session_state.my_watchlist.remove(c); st.rerun()
        watch_list = {c: st.session_state.stock_name_cache.get(c, c) for c in st.session_state.my_watchlist}
    else:
        sectors = get_all_sectors()
        sec = st.selectbox("板块", sectors)
        num = st.slider("数量", 3, 10, 5)
        watch_list = get_dynamic_pool(sec, limit=num)

    refresh = st.slider("刷新频率(秒)", 5, 60, 10)
    auto = st.checkbox("自动刷新", value=False)

c1, c2 = st.columns([0.8, 0.2])
with c1: st.title("⚡ MarketPulse Pro"); st.caption(f"Last update: {datetime.now().strftime('%H:%M:%S')}")
with c2: 
    if st.button("🔄 刷新", type="primary"): st.cache_data.clear(); st.rerun()

data_rows = []
progress = st.progress(0)
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
    futures = {executor.submit(process_single_stock_fast, c, n): c for c, n in watch_list.items()}
    for i, f in enumerate(concurrent.futures.as_completed(futures)):
        res = f.result()
        if res: data_rows.append(res)
        progress.progress((i+1)/len(watch_list))
progress.empty()

if data_rows:
    df_display = pd.DataFrame(data_rows)
    st.markdown("### 📋 实时监控列表")
    
    # 调整列宽比例以适应更多指标
    h_cols = st.columns([1.2, 0.8, 0.8, 0.6, 0.6, 0.6, 1.2, 0.8])
    headers = ["股票名称", "最新价", "涨跌幅", "趋势", "MACD", "KDJ", "走势(4H)", "操作"]
    for col, h in zip(h_cols, headers): col.markdown(f"**{h}**")
    st.divider()

    for idx, row in df_display.iterrows():
        c_cols = st.columns([1.2, 0.8, 0.8, 0.6, 0.6, 0.6, 1.2, 0.8])
        
        with c_cols[0]: st.markdown(f"**{row['名称']}**"); st.caption(f"{row['代码']}")
        with c_cols[1]: 
            color = "#d9001b" if row['今日涨跌'] > 0 else ("#00a854" if row['今日涨跌'] < 0 else "#333")
            st.markdown(f"<span style='color:{color}; font-size:1.1em; font-weight:600'>{row['最新价']:.2f}</span>", unsafe_allow_html=True)
        with c_cols[2]:
            bg = "badge-up" if row['今日涨跌'] > 0 else ("badge-down" if row['今日涨跌'] < 0 else "")
            sign = "+" if row['今日涨跌'] > 0 else ""
            st.markdown(f"<span class='{bg}'>{sign}{row['今日涨跌']*100:.2f}%</span>", unsafe_allow_html=True)
        
        # 指标展示
        with c_cols[3]:
            t_class = "tag-trend-bull" if "多" in row['趋势'] else "tag-trend-bear"
            st.markdown(f"<span class='{t_class}'>{row['趋势']}</span>", unsafe_allow_html=True)
        with c_cols[4]:
            m_class = "tag-signal-gold" if "金" in row['MACD信号'] else ("tag-signal-death" if "死" in row['MACD信号'] else "")
            st.markdown(f"<span class='{m_class}'>{row['MACD信号']}</span>", unsafe_allow_html=True)
        with c_cols[5]: st.caption(row['KDJ信号'])
            
        with c_cols[6]: st.line_chart(row['近期走势'], height=30)
        with c_cols[7]:
            if st.button("📊 分析", key=f"btn_{row['代码']}", use_container_width=True):
                open_stock_modal(row['代码'], row['名称'])
        st.markdown("<hr style='margin: 5px 0; opacity: 0.5;'>", unsafe_allow_html=True)

if auto: time.sleep(refresh); st.rerun()
