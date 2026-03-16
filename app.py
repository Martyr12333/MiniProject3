"""
Streamlit Chat Interface for FinTech Agents
Wraps the Single-Agent and Multi-Agent architectures from the notebook.
"""

import os, json, time, sqlite3, requests
import pandas as pd
import yfinance as yf
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

# ── Paths ────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "stocks.db")
CSV_PATH = os.path.join(BASE_DIR, "sp500_companies.csv")

load_dotenv(os.path.join(BASE_DIR, ".env"))

def _get_secret(key, default=""):
    """Read from st.secrets (Streamlit Cloud) → env var → default."""
    try:
        return st.secrets[key]
    except Exception:
        return os.getenv(key, default)

ALPHAVANTAGE_API_KEY = _get_secret("ALPHAVANTAGE_API_KEY", "V81MWZ10Q78DTNC3")
OPENAI_API_KEY = _get_secret("OPENAI_API_KEY")


# ── Database bootstrap ───────────────────────────────────────
def ensure_database():
    if os.path.exists(DB_PATH):
        return
    if not os.path.exists(CSV_PATH):
        st.error(
            f"Cannot find `sp500_companies.csv` at `{CSV_PATH}`. "
            "Please place it next to this app before running."
        )
        st.stop()
    df = pd.read_csv(CSV_PATH)
    df.columns = df.columns.str.strip().str.lower()
    df = df.rename(columns={
        "symbol": "ticker", "shortname": "company",
        "sector": "sector", "industry": "industry",
        "exchange": "exchange", "marketcap": "market_cap_raw",
    })

    def cap_bucket(v):
        try:
            v = float(v)
            if v >= 10_000_000_000:
                return "Large"
            return "Mid" if v >= 2_000_000_000 else "Small"
        except Exception:
            return "Unknown"

    df["market_cap"] = df["market_cap_raw"].apply(cap_bucket)
    df = (
        df.dropna(subset=["ticker", "company"])
        .drop_duplicates(subset=["ticker"])[
            ["ticker", "company", "sector", "industry", "market_cap", "exchange"]
        ]
    )
    conn = sqlite3.connect(DB_PATH)
    df.to_sql("stocks", conn, if_exists="replace", index=False)
    conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_ticker ON stocks(ticker)")
    conn.commit()
    conn.close()


# ── 7 Tool Functions ─────────────────────────────────────────
def get_price_performance(tickers: list, period: str = "1y") -> dict:
    results = {}
    for ticker in tickers:
        try:
            data = yf.download(ticker, period=period, progress=False, auto_adjust=True)
            if data.empty:
                results[ticker] = {"error": "No data — possibly delisted"}
                continue
            start = float(data["Close"].iloc[0].item())
            end = float(data["Close"].iloc[-1].item())
            results[ticker] = {
                "start_price": round(start, 2),
                "end_price": round(end, 2),
                "pct_change": round((end - start) / start * 100, 2),
                "period": period,
            }
        except Exception as e:
            results[ticker] = {"error": str(e)}
    return results


def get_market_status() -> dict:
    return requests.get(
        f"https://www.alphavantage.co/query?function=MARKET_STATUS"
        f"&apikey={ALPHAVANTAGE_API_KEY}",
        timeout=10,
    ).json()


def get_top_gainers_losers() -> dict:
    return requests.get(
        f"https://www.alphavantage.co/query?function=TOP_GAINERS_LOSERS"
        f"&apikey={ALPHAVANTAGE_API_KEY}",
        timeout=10,
    ).json()


def get_news_sentiment(ticker: str, limit: int = 5) -> dict:
    data = requests.get(
        f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT"
        f"&tickers={ticker}&limit={limit}&apikey={ALPHAVANTAGE_API_KEY}",
        timeout=10,
    ).json()
    return {
        "ticker": ticker,
        "articles": [
            {
                "title": a.get("title"),
                "source": a.get("source"),
                "sentiment": a.get("overall_sentiment_label"),
                "score": a.get("overall_sentiment_score"),
            }
            for a in data.get("feed", [])[:limit]
        ],
    }


def get_company_overview(ticker: str) -> dict:
    url = (
        f"https://www.alphavantage.co/query?function=OVERVIEW"
        f"&symbol={ticker}&apikey={ALPHAVANTAGE_API_KEY}"
    )
    response = requests.get(url, timeout=10).json()
    if "Name" not in response:
        return {"error": f"No overview data for {ticker}"}
    return {
        "ticker": ticker,
        "name": response.get("Name"),
        "sector": response.get("Sector"),
        "pe_ratio": response.get("PERatio"),
        "eps": response.get("EPS"),
        "market_cap": response.get("MarketCapitalization"),
        "52w_high": response.get("52WeekHigh"),
        "52w_low": response.get("52WeekLow"),
    }


def get_tickers_by_sector(sector: str) -> dict:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        "SELECT ticker, company, industry FROM stocks WHERE LOWER(sector) = LOWER(?)",
        conn,
        params=(sector,),
    )
    if df.empty:
        df = pd.read_sql_query(
            "SELECT ticker, company, industry FROM stocks WHERE LOWER(industry) LIKE ?",
            conn,
            params=(f"%{sector.lower()}%",),
        )
    conn.close()
    return {"sector": sector, "stocks": df.to_dict(orient="records")}


def query_local_db(sql: str) -> dict:
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query(sql, conn)
        conn.close()
        return {"columns": list(df.columns), "rows": df.to_dict(orient="records")}
    except Exception as e:
        return {"error": str(e)}


ALL_TOOL_FUNCTIONS = {
    "get_tickers_by_sector": get_tickers_by_sector,
    "get_price_performance": get_price_performance,
    "get_company_overview": get_company_overview,
    "get_market_status": get_market_status,
    "get_top_gainers_losers": get_top_gainers_losers,
    "get_news_sentiment": get_news_sentiment,
    "query_local_db": query_local_db,
}


# ── Tool Schemas ─────────────────────────────────────────────
def _s(name, desc, props, req):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": desc,
            "parameters": {"type": "object", "properties": props, "required": req},
        },
    }


SCHEMA_TICKERS = _s(
    "get_tickers_by_sector",
    "Return all stocks in a sector or industry from the local database. "
    "Use broad sector names ('Information Technology', 'Energy') or sub-sectors ('semiconductor', 'insurance').",
    {"sector": {"type": "string", "description": "Sector or industry name"}},
    ["sector"],
)
SCHEMA_PRICE = _s(
    "get_price_performance",
    "Get % price change for a list of tickers over a time period. "
    "Periods: '1mo','3mo','6mo','ytd','1y'.",
    {
        "tickers": {"type": "array", "items": {"type": "string"}},
        "period": {"type": "string", "default": "1y"},
    },
    ["tickers"],
)
SCHEMA_OVERVIEW = _s(
    "get_company_overview",
    "Get fundamentals for one stock: P/E ratio, EPS, market cap, 52-week high and low.",
    {"ticker": {"type": "string", "description": "Ticker symbol e.g. 'AAPL'"}},
    ["ticker"],
)
SCHEMA_STATUS = _s(
    "get_market_status",
    "Check whether global stock exchanges are currently open or closed.",
    {},
    [],
)
SCHEMA_MOVERS = _s(
    "get_top_gainers_losers",
    "Get today's top gaining, top losing, and most actively traded stocks.",
    {},
    [],
)
SCHEMA_NEWS = _s(
    "get_news_sentiment",
    "Get latest news headlines and Bullish/Bearish/Neutral sentiment scores for a stock.",
    {"ticker": {"type": "string"}, "limit": {"type": "integer", "default": 5}},
    ["ticker"],
)
SCHEMA_SQL = _s(
    "query_local_db",
    "Run a SQL SELECT on stocks.db. "
    "Table 'stocks': ticker, company, sector, industry, market_cap (Large/Mid/Small), exchange.",
    {"sql": {"type": "string", "description": "A valid SQL SELECT statement"}},
    ["sql"],
)

ALL_SCHEMAS = [
    SCHEMA_TICKERS, SCHEMA_PRICE, SCHEMA_OVERVIEW,
    SCHEMA_STATUS, SCHEMA_MOVERS, SCHEMA_NEWS, SCHEMA_SQL,
]

DB_TOOLS = [SCHEMA_TICKERS, SCHEMA_SQL]
DATA_TOOLS = [SCHEMA_PRICE, SCHEMA_OVERVIEW, SCHEMA_STATUS, SCHEMA_MOVERS, SCHEMA_NEWS]


# ── Confidence scoring ────────────────────────────────────────
def _compute_confidence(answer, tools_called, raw_data, tool_schemas, reached_max):
    """Score confidence 0.0-1.0 based on tool execution quality."""
    issues = []
    if reached_max:
        issues.append("Hit max iterations without final answer")
        return 0.20, issues
    if not answer or not answer.strip():
        issues.append("Empty answer")
        return 0.10, issues

    score = 1.0

    error_count = 0
    for name, data in raw_data.items():
        if isinstance(data, dict) and "error" in data:
            error_count += 1
            issues.append(f"Tool '{name}' returned an error")
        elif isinstance(data, dict):
            for v in data.values():
                if isinstance(v, dict) and "error" in v:
                    error_count += 1
                    issues.append(f"Tool '{name}' had partial errors")
                    break
    if error_count:
        score -= 0.15 * error_count

    if tool_schemas and not tools_called:
        score -= 0.10
        issues.append("No tools called despite tools being available")

    lower = answer.lower()
    failure_phrases = ["unable to retrieve", "could not retrieve", "data not available",
                       "no data found", "i don't have access", "cannot find any"]
    hits = [p for p in failure_phrases if p in lower]
    if hits:
        score -= 0.15 * min(len(hits), 2)
        issues.append(f"Answer indicates data retrieval failure: {', '.join(hits[:3])}")

    return max(round(score, 2), 0.05), issues


# ══════════════════════════════════════════════════════════════
#  Agent loops with conversational memory
# ══════════════════════════════════════════════════════════════

SINGLE_AGENT_PROMPT = (
    "You are a highly capable FinTech AI with access to 7 live data tools.\n\n"
    "For EVERY question, internally follow the Plan-and-Solve protocol:\n\n"
    "1. UNDERSTAND: Parse the question. Identify the entities (tickers, sectors), "
    "the metrics requested (price, P/E, sentiment), and the time period.\n\n"
    "2. PLAN: Before calling any tool, decide your plan. For example: "
    "look up Energy sector tickers via get_tickers_by_sector, "
    "fetch 1-year price performance for each, rank by pct_change.\n\n"
    "3. EXECUTE: Carry out the plan by calling tools in the order you planned. "
    "Chain tools when needed — first resolve tickers (query_local_db / get_tickers_by_sector), "
    "then fetch data (get_price_performance / get_company_overview / get_news_sentiment).\n\n"
    "4. VERIFY & ANSWER: Review all tool outputs. Check for errors or missing data. "
    "Then provide a clear, data-backed final answer.\n\n"
    "CRITICAL RULES:\n"
    "- Never fabricate data. If a tool returns an error, say so.\n"
    "- Do NOT expose your internal reasoning steps (Step 1/2/3/4, Plan, etc.) in "
    "your final answer to the user. Only present the final, polished answer.\n"
    "- Use conversation history to resolve pronouns and follow-up references "
    "(e.g. 'that company', 'how does it compare', 'which of the two').\n"
    "- When the user asks a follow-up, do NOT repeat the full prior answer — "
    "answer the new question directly, referencing prior context as needed.\n"
)

MULTI_AGENT_ORCHESTRATOR_CONTEXT = (
    "Below is the conversation so far between the user and the assistant. "
    "Use this context to understand references like 'that stock', 'compare to the previous one', etc.\n\n"
)


def _build_history_messages(conversation_pairs):
    """Convert stored (user, assistant) pairs into OpenAI message format."""
    msgs = []
    for user_text, asst_text in conversation_pairs:
        msgs.append({"role": "user", "content": user_text})
        msgs.append({"role": "assistant", "content": asst_text})
    return msgs


def _build_history_text(conversation_pairs):
    """Build a plain-text summary of conversation history for multi-agent context."""
    if not conversation_pairs:
        return ""
    lines = []
    for i, (u, a) in enumerate(conversation_pairs, 1):
        lines.append(f"[Turn {i}] User: {u}")
        lines.append(f"[Turn {i}] Assistant: {a}")
    return "\n".join(lines)


def run_single_agent_with_memory(client, model, question, conversation_pairs):
    """
    Single-agent loop that carries conversation history as prior messages.
    Returns (answer_text, tools_called_list, confidence, issues).
    """
    messages = [{"role": "system", "content": SINGLE_AGENT_PROMPT}]
    messages.extend(_build_history_messages(conversation_pairs))
    messages.append({"role": "user", "content": question})

    tools_called = []
    raw_data = {}

    for _ in range(20):
        kwargs = {"model": model, "messages": messages, "tools": ALL_SCHEMAS}
        resp = client.chat.completions.create(**kwargs)
        msg = resp.choices[0].message
        messages.append(msg)

        if not msg.tool_calls:
            answer = msg.content or ""
            conf, issues = _compute_confidence(answer, tools_called, raw_data, ALL_SCHEMAS, reached_max=False)
            return answer, tools_called, conf, issues

        for tc in msg.tool_calls:
            fname = tc.function.name
            tools_called.append(fname)
            args = json.loads(tc.function.arguments)
            result = ALL_TOOL_FUNCTIONS[fname](**args)
            raw_data[fname] = result
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "name": fname,
                "content": json.dumps(result),
            })

    conf, issues = _compute_confidence("", tools_called, raw_data, ALL_SCHEMAS, reached_max=True)
    return "Max iterations reached without a final answer.", tools_called, conf, issues


def _run_specialist(client, model, agent_name, system_prompt, task, tool_schemas, max_iters=8):
    """Stateless specialist loop (used inside multi-agent pipeline).
    Returns (answer, tools_called, confidence, issues).
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": task},
    ]
    tools_called = []
    raw_data = {}

    for _ in range(max_iters):
        kwargs = {"model": model, "messages": messages}
        if tool_schemas:
            kwargs["tools"] = tool_schemas
        resp = client.chat.completions.create(**kwargs)
        msg = resp.choices[0].message
        messages.append(msg)

        if not msg.tool_calls:
            answer = msg.content or ""
            conf, issues = _compute_confidence(answer, tools_called, raw_data, tool_schemas, reached_max=False)
            return answer, tools_called, conf, issues

        for tc in msg.tool_calls:
            fname = tc.function.name
            tools_called.append(fname)
            args = json.loads(tc.function.arguments)
            result = ALL_TOOL_FUNCTIONS[fname](**args)
            raw_data[fname] = result
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "name": fname,
                "content": json.dumps(result),
            })

    conf, issues = _compute_confidence("", tools_called, raw_data, tool_schemas, reached_max=True)
    return "Max iterations reached.", tools_called, conf, issues


def run_multi_agent_with_memory(client, model, question, conversation_pairs):
    """
    Sequential-pipeline multi-agent: DB Agent -> Data Agent -> Synthesizer.
    Conversation history is injected as textual context.
    Returns (answer_text, tools_called_list, agents_activated, confidence, issue_count).
    """
    history_text = _build_history_text(conversation_pairs)
    context_block = ""
    if history_text:
        context_block = (
            f"{MULTI_AGENT_ORCHESTRATOR_CONTEXT}"
            f"{history_text}\n\n---\n\n"
        )

    sys1 = (
        "You are the Database Specialist. Internally follow this protocol:\n"
        "1. UNDERSTAND: Identify which tickers, sectors, or industries the question refers to.\n"
        "2. PLAN: Decide whether to use get_tickers_by_sector (broad sector lookup) "
        "or query_local_db (specific SQL).\n"
        "3. EXECUTE: Run your planned queries.\n"
        "4. VERIFY: Confirm you found relevant tickers. If none found, try an alternative query.\n\n"
        "Do NOT expose step labels or internal reasoning in your response.\n"
        "IMPORTANT: The `stocks` table ONLY has columns: ticker, company, sector, industry, "
        "market_cap, exchange. Do NOT query for price or returns in SQL. "
        "Use conversation history to resolve any follow-up references."
    )
    task1 = f"{context_block}Current question: {question}"
    answer1, tools1, conf1, issues1 = _run_specialist(client, model, "DB Agent", sys1, task1, DB_TOOLS, 4)

    sys2 = (
        "You are the Data Fetcher. Internally follow this protocol:\n"
        "1. UNDERSTAND: Read the original question and the tickers provided. "
        "Identify what data is needed (prices, fundamentals, news, market status).\n"
        "2. PLAN: Decide which tools to call and in what order.\n"
        "3. EXECUTE: Call the tools as planned.\n"
        "4. VERIFY: Check that all requested data was fetched. Report any errors encountered.\n\n"
        "Do NOT expose step labels or internal reasoning in your response.\n"
        "Use conversation history to understand what data is being asked for."
    )
    task2 = f"{context_block}Current question: {question}\n\nTickers identified:\n{answer1}"
    answer2, tools2, conf2, issues2 = _run_specialist(client, model, "Data Agent", sys2, task2, DATA_TOOLS, 6)

    sys3 = (
        "You are the Synthesizer. Internally follow this protocol:\n"
        "1. UNDERSTAND: Re-read the original question to know exactly what is being asked.\n"
        "2. PLAN: Outline how to structure your answer (e.g., ranking, comparison table, summary).\n"
        "3. EXECUTE: Write the answer using ONLY the collected data. Do NOT fabricate numbers.\n"
        "4. VERIFY: Cross-check your answer against the data. Flag any gaps or missing information.\n\n"
        "Do NOT expose step labels or internal reasoning in your response. "
        "Only present the final, polished answer.\n"
        "Use conversation history for context on follow-up questions."
    )
    task3 = f"{context_block}Current question: {question}\n\nCollected Data:\n{answer2}"
    answer3, _, conf3, issues3 = _run_specialist(client, model, "Synthesizer", sys3, task3, [], 2)

    # Pipeline-level verification: adjust confidence based on upstream quality
    pipeline_penalty = 0.0
    serious_keywords = ["error", "returned an error", "partial errors", "retrieval failure"]
    upstream_errors = [iss for iss in issues1 + issues2
                       if any(k in iss.lower() for k in serious_keywords)]
    if upstream_errors:
        pipeline_penalty += 0.10

    lower = answer3.lower()
    failure_phrases = ["unable to retrieve", "could not retrieve", "data not available",
                       "no data found", "cannot find any"]
    if any(p in lower for p in failure_phrases):
        pipeline_penalty += 0.15

    final_confidence = max(round(conf3 - pipeline_penalty, 2), 0.05)
    all_issues = issues1 + issues2 + issues3

    all_tools = tools1 + tools2
    agents = ["DB Agent", "Data Agent", "Synthesizer"]
    return answer3, all_tools, agents, final_confidence, len(all_issues)


# ══════════════════════════════════════════════════════════════
#  Streamlit UI
# ══════════════════════════════════════════════════════════════

st.set_page_config(page_title="FinTech Agent Chat", page_icon="📈", layout="wide")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "conversation_pairs" not in st.session_state:
    st.session_state.conversation_pairs = []

# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")

    agent_mode = st.radio(
        "Agent Architecture",
        ["Single Agent", "Multi-Agent"],
        index=0,
    )

    model_choice = st.radio(
        "Model",
        ["gpt-4o-mini", "gpt-4o"],
        index=0,
    )

    st.divider()

    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.conversation_pairs = []
        st.rerun()

    st.divider()
    st.caption(
        f"Architecture: **{agent_mode}**\n\n"
        f"Model: **{model_choice}**\n\n"
        f"Turns: **{len(st.session_state.conversation_pairs)}**"
    )

# ── Validate prerequisites ───────────────────────────────────
if not OPENAI_API_KEY:
    st.warning("OpenAI API Key not found. Please set it in Streamlit Secrets or as an environment variable.")
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY)
ensure_database()

# ── Main chat area ───────────────────────────────────────────
st.title("📈 FinTech Agent Chat")
st.caption("Ask the AI financial assistant anything — supports multi-turn follow-up questions")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("meta"):
            m = msg["meta"]
            cols = st.columns(4)
            cols[0].caption(f"🤖 {m.get('agent_type', '')}")
            cols[1].caption(f"🧠 {m.get('model', '')}")
            cols[2].caption(f"📊 Confidence: {m.get('confidence', 0):.0%}")
            if m.get("tools_used"):
                cols[3].caption(f"🔧 {', '.join(m['tools_used'])}")

# ── Handle new input ─────────────────────────────────────────
if prompt := st.chat_input("Ask a question…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            t0 = time.time()
            try:
                if agent_mode == "Single Agent":
                    answer, tools_used, confidence, issues = run_single_agent_with_memory(
                        client, model_choice, prompt,
                        st.session_state.conversation_pairs,
                    )
                    agents_activated = ["Single Agent"]
                    issue_count = len(issues)
                else:
                    answer, tools_used, agents_activated, confidence, issue_count = run_multi_agent_with_memory(
                        client, model_choice, prompt,
                        st.session_state.conversation_pairs,
                    )
                elapsed = round(time.time() - t0, 1)
            except requests.exceptions.ConnectionError:
                answer = (
                    "⚠️ Cannot connect to Alpha Vantage API. "
                    "Please check your network connection and API key."
                )
                tools_used = []
                agents_activated = []
                confidence = 0.0
                issue_count = 0
                elapsed = 0

        st.markdown(answer)

        meta = {
            "agent_type": agent_mode,
            "model": model_choice,
            "tools_used": list(dict.fromkeys(tools_used)),
            "agents": agents_activated,
            "elapsed": elapsed,
            "confidence": confidence,
            "issue_count": issue_count,
        }

        cols = st.columns(4)
        cols[0].caption(f"🤖 {agent_mode}")
        cols[1].caption(f"🧠 {model_choice}")
        cols[2].caption(f"📊 Confidence: {confidence:.0%}")
        if tools_used:
            cols[3].caption(f"🔧 {', '.join(dict.fromkeys(tools_used))}")

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "meta": meta,
    })
    st.session_state.conversation_pairs.append((prompt, answer))
