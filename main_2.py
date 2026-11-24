import os
import requests
from functools import lru_cache
import time
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv
from google import genai
from google.genai import types
import json

# -------------------------------------------------
# SETUP
# -------------------------------------------------

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError(
        "GEMINI_API_KEY was not found. "
        "Check that your .env file is in the same folder as main.py."
    )

client = genai.Client(api_key=api_key)

# -------------------------------------------------
# HELPER FUNCTIONS
# -------------------------------------------------

def clean_number(value):
    """
    Convert pandas/numpy values into normal Python numbers.
    """

    if value is None:
        return None

    try:
        if value != value:  # catches NaN
            return None
    except Exception:
        pass

    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def format_date(date_value):
    """
    Convert a pandas/yfinance date into a readable DD-MM-YYYY string.
    """

    if date_value is None:
        return None

    try:
        return date_value.strftime("%d-%m-%Y")
    except AttributeError:
        return str(date_value)


def get_latest_period(statement):
    """
    Return only the fiscal year rather than an exact date,
    as yfinance dates are not always exact fiscal year-end dates.
    """

    if statement is None or statement.empty:
        return None

    if len(statement.columns) == 0:
        return None

    try:
        return str(statement.columns[0].year)
    except Exception:
        return str(statement.columns[0])


def get_latest_value(statement, row_names):
    """
    Search one or more possible row names and return
    the latest non-missing value.
    """

    if statement is None or statement.empty:
        return None

    if isinstance(row_names, str):
        row_names = [row_names]

    for row_name in row_names:

        if row_name in statement.index:

            values = statement.loc[row_name]

            for value in values:

                cleaned_value = clean_number(value)

                if cleaned_value is not None:
                    return cleaned_value

    return None


def safe_divide(numerator, denominator):
    """
    Divide two numbers safely.
    """

    if numerator is None or denominator is None:
        return None

    if denominator == 0:
        return None

    return numerator / denominator

def to_significant_figures(value, sig_figs=3):
    """
    Format a number to a given number of significant figures.

    Returns a string. Used only for display — internal calculations
    always use the full-precision numeric value.
    """

    if value is None:
        return None

    try:
        value = float(value)
    except (TypeError, ValueError):
        return value

    if value == 0:
        return "0"

    from math import floor, log10

    exponent = floor(log10(abs(value)))

    return f"{round(value, sig_figs - 1 - exponent):.{max(0, sig_figs - 1 - exponent)}f}"

# -------------------------------------------------
# CACHED DATA RETRIEVAL
# -------------------------------------------------

@lru_cache(maxsize=20)
def get_company_data(ticker: str):
    """
    Download all required yfinance data once for a ticker.

    Subsequent tools reuse the cached result rather than
    making repeated requests.
    """

    ticker = ticker.strip().upper()

    print(f"\n[Downloading yfinance data for {ticker}]")

    company = yf.Ticker(ticker)

    return {
        "ticker": ticker,
        "info": company.info,
        "income_statement": company.income_stmt,
        "balance_sheet": company.balance_sheet,
        "cash_flow": company.cashflow
    }


# -------------------------------------------------
# TOOL 1: COMPANY PROFILE
# -------------------------------------------------

def get_company_profile(ticker: str) -> dict:
    """
    Get basic factual information about a public company.

    Args:
        ticker: Stock ticker symbol, for example AAPL.

    Returns:
        Company name, sector, industry, country,
        business description and market capitalisation.
    """

    print("[Tool used: company profile]")

    data = get_company_data(ticker)
    info = data["info"]

    return {
        "ticker": data["ticker"],
        "company_name": info.get("longName"),
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "country": info.get("country"),
        "market_cap": clean_number(
            info.get("marketCap")
        ),
        "business_summary": info.get(
            "longBusinessSummary"
        )
    }


# -------------------------------------------------
# TOOL 2: INCOME STATEMENT
# -------------------------------------------------

def get_income_statement(ticker: str) -> dict:
    """
    Get important figures from the latest available
    annual income statement.

    Args:
        ticker: Stock ticker symbol.

    Returns:
        Latest fiscal period plus revenue,
        gross profit, EBIT, EBITDA and net income.
    """

    print("[Tool used: income statement]")

    data = get_company_data(ticker)

    statement = data["income_statement"]

    return {
        "fiscal_period": get_latest_period(statement),

        "revenue": get_latest_value(
            statement,
            ["Total Revenue", "Operating Revenue"]
        ),

        "gross_profit": get_latest_value(
            statement,
            "Gross Profit"
        ),

        "operating_income": get_latest_value(
            statement,
            [
                "Operating Income",
                "Total Operating Income As Reported"
            ]
        ),

        "ebit": get_latest_value(
            statement,
            "EBIT"
        ),

        "ebitda": get_latest_value(
            statement,
            ["EBITDA", "Normalized EBITDA"]
        ),

        "net_income": get_latest_value(
            statement,
            [
                "Net Income",
                "Net Income Common Stockholders"
            ]
        ),

        "interest_expense": get_latest_value(
            statement,
            [
                "Interest Expense",
                "Interest Expense Non Operating"
            ]
        )
    }


# -------------------------------------------------
# TOOL 3: BALANCE SHEET
# -------------------------------------------------

def get_balance_sheet(ticker: str) -> dict:
    """
    Get important figures from the latest available
    annual balance sheet.

    Args:
        ticker: Stock ticker symbol.

    Returns:
        Latest fiscal period plus cash, debt,
        assets, liabilities and equity.
    """

    print("[Tool used: balance sheet]")

    data = get_company_data(ticker)

    statement = data["balance_sheet"]

    return {
        "fiscal_period": get_latest_period(statement),

        "cash_and_short_term_investments":
            get_latest_value(
                statement,
                [
                    "Cash Cash Equivalents And Short Term Investments",
                    "Cash And Cash Equivalents"
                ]
            ),

        "total_debt":
            get_latest_value(
                statement,
                "Total Debt"
            ),

        "current_assets":
            get_latest_value(
                statement,
                "Current Assets"
            ),

        "current_liabilities":
            get_latest_value(
                statement,
                "Current Liabilities"
            ),

        "total_assets":
            get_latest_value(
                statement,
                "Total Assets"
            ),

        "total_liabilities":
            get_latest_value(
                statement,
                "Total Liabilities Net Minority Interest"
            ),

        "stockholders_equity":
            get_latest_value(
                statement,
                [
                    "Stockholders Equity",
                    "Total Equity Gross Minority Interest"
                ]
            )
    }


# -------------------------------------------------
# TOOL 4: CASH FLOW
# -------------------------------------------------

def get_cash_flow(ticker: str) -> dict:
    """
    Get important figures from the latest available
    annual cash-flow statement.

    Args:
        ticker: Stock ticker symbol.

    Returns:
        Latest fiscal period plus operating cash flow,
        capital expenditure and free cash flow.
    """

    print("[Tool used: cash flow statement]")

    data = get_company_data(ticker)

    statement = data["cash_flow"]

    return {
        "fiscal_period": get_latest_period(statement),

        "operating_cash_flow":
            get_latest_value(
                statement,
                [
                    "Operating Cash Flow",
                    "Total Cash From Operating Activities"
                ]
            ),

        "capital_expenditure":
            get_latest_value(
                statement,
                [
                    "Capital Expenditure",
                    "Capital Expenditures"
                ]
            ),

        "free_cash_flow":
            get_latest_value(
                statement,
                "Free Cash Flow"
            )
    }


# -------------------------------------------------
# TOOL 5: BASELINE RATIOS
# -------------------------------------------------

def calculate_baseline_ratios(
    income: dict,
    balance: dict,
    cash_flow: dict
) -> dict:
    """
    Calculate baseline financial ratios using
    deterministic Python arithmetic.

    Args:
        income: Retrieved income statement data.
        balance: Retrieved balance sheet data.
        cash_flow: Retrieved cash flow data.

    Returns:
        Profitability, liquidity and leverage ratios.
    """

    print("[Tool used: baseline ratio calculator]")

    revenue = income["revenue"]
    operating_income = income["operating_income"]
    net_income = income["net_income"]
    ebit = income["ebit"]
    ebitda = income["ebitda"]
    interest_expense = income["interest_expense"]

    cash = balance[
        "cash_and_short_term_investments"
    ]

    debt = balance["total_debt"]

    current_assets = balance[
        "current_assets"
    ]

    current_liabilities = balance[
        "current_liabilities"
    ]

    free_cash_flow = cash_flow[
        "free_cash_flow"
    ]

    net_debt = None

    if debt is not None and cash is not None:
        net_debt = debt - cash

    if interest_expense is not None:
        interest_expense = abs(interest_expense)

    return {
        "income_statement_period":
            income["fiscal_period"],

        "balance_sheet_period":
            balance["fiscal_period"],

        "cash_flow_period":
            cash_flow["fiscal_period"],

        "operating_margin":
            safe_divide(
                operating_income,
                revenue
            ),

        "net_profit_margin":
            safe_divide(
                net_income,
                revenue
            ),

        "free_cash_flow_margin":
            safe_divide(
                free_cash_flow,
                revenue
            ),

        "current_ratio":
            safe_divide(
                current_assets,
                current_liabilities
            ),

        "net_debt":
            net_debt,

        "net_debt_to_ebitda":
            safe_divide(
                net_debt,
                ebitda
            ),

        "interest_coverage_ratio":
            safe_divide(
                ebit,
                interest_expense
            )
    }


# -------------------------------------------------
# RESPONSE TEXT EXTRACTION
# -------------------------------------------------

def extract_text(response):
    """
    Extract only visible text parts from Gemini's response.

    This avoids the thought_signature warning that can appear
    when using response.text.
    """

    text_parts = []

    if not response.candidates:
        return ""

    content = response.candidates[0].content

    if content is None:
        return ""

    for part in content.parts:

        if getattr(part, "text", None):
            text_parts.append(part.text)

    return "\n".join(text_parts)


# -------------------------------------------------
# STAGE 2 AGENT
# -------------------------------------------------

def run_financial_agent(ticker, company_context):

    prompt = f"""
You are the baseline financial analysis component of a Financial Resilience Agent.

Company ticker: {ticker}

Use ONLY the company data supplied below.

Do not call or request any additional tools.
Do not invent missing company-specific facts.
Do not use prior knowledge about the company.

Company data:

{json.dumps(company_context, indent=2)}

Prepare a concise baseline financial assessment covering:

1. Company overview
2. Sector and industry
3. Fiscal period used
4. Revenue and profitability
5. Cash-flow generation
6. Liquidity
7. Debt and leverage
8. Important baseline financial ratios
9. Short interpretation of financial strengths and vulnerabilities

Rules:
- Use the supplied Python-calculated ratios.
- Do not recalculate ratios yourself.
- State fiscal year only.
- Margins should be shown as percentages.
- Leverage and coverage ratios should be shown as multiples.
- Base all interpretations directly on the supplied financial data.
"""

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.2
        )
    )

    return extract_text(response)


# -------------------------------------------------
# STAGE 3A — SCENARIO GENERATION
# -------------------------------------------------

def run_scenario_agent(ticker, company_context):

    prompt = f"""
You are the stress-scenario generation component of a financial resilience analysis.

Company ticker: {ticker}

Use ONLY the company information supplied below.

Company information:

{json.dumps(company_context, indent=2)}

Generate exactly 3 distinct and economically plausible stress scenarios for this company.

For each scenario:
- identify the economic event
- identify the underlying cause
- identify the most material transmission mechanisms
- normally use 2–4 mechanisms
- focus only on mechanisms likely to have a substantial financial impact
- identify the relevant company exposure
- identify the financial variables affected
- identify whether each variable would increase or decrease

Additionally, for each scenario identify a SINGLE macro driver:
- variable: one of
  "GDP", "policy_rate", "oil_price", "CPI", "USD_index", "credit_spread"
- adverse_direction: "increase" or "decrease" — the direction of the
  variable that is HARMFUL to this company.

Rules for the macro driver:
- Choose the single variable whose adverse movement most directly drives
  the scenario. If two variables apply, pick the more fundamental one.
- adverse_direction is from the COMPANY's perspective: if a rate hike
  hurts this company, adverse_direction is "increase"; if a rate cut
  hurts, adverse_direction is "decrease".

Important rules:
- Do not call or request any additional tools.
- Do not quantify any financial impact yet.
- Do not assign percentage shocks.
- Do not invent company-specific facts that are not supported by the supplied data.
- Do not use prior knowledge about the company.
- Keep the 3 scenarios materially distinct from one another.

Return ONLY valid JSON in this structure:

{{
  "scenarios": [
    {{
      "scenario_name": "...",
      "economic_event": "...",
      "underlying_cause": "...",
      "macro_driver": {{
          "variable": "GDP",
          "adverse_direction": "decrease"
      }},
      "transmission_mechanisms": [
        {{
          "mechanism": "...",
          "company_exposure": "...",
          "financial_effects": [
            {{
              "variable": "...",
              "direction": "increase/decrease"
            }}
          ]
        }}
      ]
    }}
  ]
}}
"""

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.3,
            response_mime_type="application/json",
        ),
    )

    text = extract_text(response)

    if not text:
        raise ValueError("Gemini returned no scenario output.")

    return json.loads(text)


# -------------------------------------------------
# SEARCH OPENALEX
# -------------------------------------------------

def reconstruct_abstract(inverted_index):
    """
    OpenAlex delivers abstracts as an inverted index mapping each word
    to the list of positions where it appears. Reconstruct the plain
    abstract string.
    """

    if not inverted_index:
        return None

    positions = []

    for word, pos_list in inverted_index.items():
        for pos in pos_list:
            positions.append((pos, word))

    positions.sort()

    return " ".join(word for _, word in positions)


def clean_paper_results(papers: list) -> list:
    """
    Normalise OpenAlex work records into our internal paper schema.
    """

    cleaned = []

    for paper in papers:

        authors = []
        for authorship in paper.get("authorships", []) or []:
            name = (authorship.get("author") or {}).get("display_name")
            if name:
                authors.append(name)

        doi_raw = paper.get("doi") or ""
        doi = doi_raw.replace("https://doi.org/", "") if doi_raw else None

        primary_location = paper.get("primary_location") or {}
        source = primary_location.get("source") or {}
        venue = source.get("display_name")

        abstract = reconstruct_abstract(
            paper.get("abstract_inverted_index")
        )

        cleaned.append({
            "paper_id": paper.get("id"),
            "title": paper.get("title") or paper.get("display_name"),
            "abstract": abstract,
            "year": paper.get("publication_year"),
            "authors": authors,
            "citation_count": paper.get("cited_by_count"),
            "venue": venue,
            "doi": doi,
            "url": paper.get("id"),
        })

    return cleaned


OPENALEX_STOPWORDS = {
    "a", "about", "against", "an", "and", "are", "as", "at",
    "be", "been", "being", "between", "by", "can", "could",
    "did", "do", "does", "doing", "done", "during",
    "for", "from", "had", "has", "have", "having", "how",
    "in", "into", "is", "it", "its",
    "of", "on", "or", "over",
    "should", "so", "that", "the", "their", "there", "these",
    "this", "those", "through", "to",
    "under", "upon",
    "was", "we", "were", "what", "when", "where", "which", "who",
    "why", "will", "with", "would",
}


def sanitize_openalex_query(query: str) -> str:
    """
    Strip punctuation, drop stopwords, and cap the query length so the
    request has a chance of matching on OpenAlex.
    """

    if not query:
        return ""

    cleaned = query

    for char in ("?", "!", '"', "'", ":", ",", ".", ";", "(", ")"):
        cleaned = cleaned.replace(char, " ")

    tokens = cleaned.lower().split()

    content = [
        token for token in tokens
        if token not in OPENALEX_STOPWORDS
    ]

    content = content[:8]

    return " ".join(content)


LITERATURE_SEARCH_LIMIT = 10
LITERATURE_TOP_K = 3
LITERATURE_RECENT_YEAR = 2010
LITERATURE_LEGACY_YEAR = 2000


def request_with_retry(url, params, max_retries=4, base_wait=2.0):
    """
    GET a URL with exponential backoff on transient server errors.
    5xx responses are retried; 4xx raise immediately.
    """

    last_response = None

    for attempt in range(max_retries + 1):

        response = requests.get(
            url, params=params, timeout=20
        )

        if response.status_code < 500:
            return response

        last_response = response

        if attempt < max_retries:
            wait = base_wait * (attempt + 1)
            print(
                f"[Transient {response.status_code} from "
                f"{url.split('?')[0]}. Retrying in {wait:.1f}s "
                f"(attempt {attempt + 1}/{max_retries})]"
            )
            time.sleep(wait)

    return last_response


@lru_cache(maxsize=100)
def search_openalex(
    query: str,
    limit: int = 6,
    year_from: int = None
) -> list:
    """
    Search OpenAlex for papers matching the query.
    """

    sanitized = sanitize_openalex_query(query)

    print(
        f"[Searching OpenAlex: {sanitized}"
        + (f" | from {year_from}" if year_from else "")
        + "]"
    )

    url = "https://api.openalex.org/works"

    params = {
        "search": sanitized,
        "per-page": limit,
    }

    if year_from:
        params["filter"] = (
            f"from_publication_date:{year_from}-01-01"
        )

    response = request_with_retry(url, params)

    response.raise_for_status()

    data = response.json()

    return clean_paper_results(
        data.get("results", [])
    )


# -------------------------------------------------
# STAGE 3B — MACRO SHOCK LOOKUP (FRED)
# -------------------------------------------------

fred_api_key = os.getenv("FRED_API_KEY")

FRED_SERIES = {
    "GDP":           {"id": "GDPC1",      "transform": "pct_change"},
    "policy_rate":   {"id": "FEDFUNDS",   "transform": "change_pp"},
    "oil_price":     {"id": "DCOILWTICO", "transform": "pct_change"},
    "CPI":           {"id": "CPIAUCSL",   "transform": "pct_change"},
    "USD_index":     {"id": "DTWEXBGS",   "transform": "pct_change"},
    "credit_spread": {"id": "BAA10Y",     "transform": "change_pp"},
}

MACRO_SHOCK_PERCENTILE = 95
MACRO_SHOCK_WINDOW_YEARS = 20


def fetch_fred_series(series_id):

    if not fred_api_key:
        raise ValueError(
            "FRED_API_KEY was not found in .env."
        )

    url = "https://api.stlouisfed.org/fred/series/observations"

    params = {
        "series_id": series_id,
        "api_key": fred_api_key,
        "file_type": "json",
    }

    response = request_with_retry(url, params)

    response.raise_for_status()

    data = response.json()

    rows = []

    for obs in data.get("observations", []):

        value_str = obs.get("value")

        if value_str in (None, "."):
            continue

        try:
            rows.append(
                (pd.Timestamp(obs["date"]), float(value_str))
            )
        except (ValueError, TypeError, KeyError):
            continue

    if not rows:
        return pd.Series(dtype=float)

    dates, values = zip(*rows)

    series = pd.Series(
        list(values),
        index=pd.DatetimeIndex(dates),
    )

    return series.sort_index()


@lru_cache(maxsize=20)
def fetch_fred_series_cached(series_id):
    return fetch_fred_series(series_id)


def fred_series_to_monthly(series):

    if series.empty:
        return series

    try:
        monthly = series.resample("ME").last()
    except (ValueError, KeyError):
        monthly = series.resample("M").last()

    return monthly.ffill()


@lru_cache(maxsize=32)
def get_macro_shock(variable, adverse_direction):

    if variable not in FRED_SERIES:
        return None

    spec = FRED_SERIES[variable]

    try:
        raw = fetch_fred_series_cached(spec["id"])
    except Exception as error:
        print(f"[FRED fetch failed for {spec['id']}: {error}]")
        return None

    if raw.empty:
        return None

    monthly = fred_series_to_monthly(raw)

    if spec["transform"] == "pct_change":
        changes = monthly.pct_change(12)
        shock_unit = "fractional_change"
    elif spec["transform"] == "change_pp":
        changes = monthly.diff(12)
        shock_unit = "percentage_point"
    else:
        return None

    cutoff = (
        monthly.index.max()
        - pd.DateOffset(years=MACRO_SHOCK_WINDOW_YEARS)
    )

    changes = changes[changes.index >= cutoff].dropna()

    if changes.empty:
        return None

    if adverse_direction == "increase":
        value = float(
            changes.quantile(MACRO_SHOCK_PERCENTILE / 100.0)
        )
    elif adverse_direction == "decrease":
        value = float(
            changes.quantile(1.0 - MACRO_SHOCK_PERCENTILE / 100.0)
        )
    else:
        return None

    return {
        "variable": variable,
        "adverse_direction": adverse_direction,
        "shock_value": value,
        "shock_unit": shock_unit,
        "percentile": MACRO_SHOCK_PERCENTILE,
        "fred_series": spec["id"],
        "history_start": str(changes.index.min().date()),
        "history_end": str(changes.index.max().date()),
        "n_observations": int(len(changes)),
    }


# -------------------------------------------------
# STAGE 3B — MECHANISM CLASSIFICATION
# -------------------------------------------------

def classify_scenario_mechanisms(scenario, company_context):

    slim_context = {
        "profile": company_context.get("profile"),
        "income_statement": company_context.get("income_statement"),
        "balance_sheet": company_context.get("balance_sheet"),
        "ratios": company_context.get("ratios"),
    }

    prompt = f"""You are the quantification classifier of a Financial Resilience Agent.

SCENARIO
{json.dumps(scenario, indent=2)}

COMPANY CONTEXT
{json.dumps(slim_context, indent=2)}

Your job: for each transmission mechanism in the scenario, decide how
it should be quantified. Three methods are possible:

1. "structural" — the effect can be computed by arithmetic from company
   data alone, with NO elasticity needed. Only the following templates
   qualify:
   - interest_rate_shock_on_debt: a change in the policy rate changes
     interest expense by total_debt × Δrate.
   - credit_spread_shock_on_debt: a change in the credit spread changes
     interest expense by total_debt × Δspread.
   No other mechanism qualifies as structural.

2. "empirical" — the effect requires an elasticity, sensitivity or
   regression coefficient from the research literature. Use this
   whenever the mechanism is of the form "revenue falls when GDP falls",
   "capex responds to rates", etc., and no arithmetic template fits.

3. "assumption" — the mechanism is real, but no plausible path to an
   elasticity exists (too idiosyncratic, or far outside the economics
   literature). Prefer "empirical" whenever literature could plausibly help.

For each mechanism, also pick the PRIMARY financial variable affected.
Choose exactly ONE variable from the mechanism's financial_effects list.
If none is suitable, pick the closest and explain why in the reasoning.

Rules:
- structural_template is filled ONLY for method="structural", and must
  be one of the two template names above.
- empirical_target_quantity is filled ONLY for method="empirical", and
  must be a concrete measurable quantity like "revenue elasticity to GDP".
- assumption_justification is filled ONLY for method="assumption", and
  must explain in one sentence why literature search is unlikely to help.
- Every mechanism must be classified. Do not skip any.

Return ONLY valid JSON in exactly this shape:

{{
  "classifications": [
    {{
      "mechanism_index": 0,
      "method": "structural",
      "reasoning": "one short sentence",
      "primary_variable": "interest_expense",
      "structural_template": "interest_rate_shock_on_debt",
      "empirical_target_quantity": null,
      "assumption_justification": null
    }}
  ]
}}
"""

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.1,
            response_mime_type="application/json",
        ),
    )

    text = extract_text(response)

    if not text:
        raise ValueError("Gemini returned no classification output.")

    return json.loads(text)


# -------------------------------------------------
# STAGE 3B — STRUCTURAL TEMPLATE
# -------------------------------------------------

def apply_structural_template(
    template_name,
    macro_shock,
    mechanism,
    company_context,
):

    balance = company_context.get("balance_sheet") or {}
    income = company_context.get("income_statement") or {}

    total_debt = balance.get("total_debt")
    interest_expense = income.get("interest_expense")

    if template_name in (
        "interest_rate_shock_on_debt",
        "credit_spread_shock_on_debt",
    ):

        if total_debt is None or total_debt == 0:
            return {
                "success": False,
                "reason": "total_debt missing or zero",
            }

        delta = macro_shock.get("shock_value")
        unit = macro_shock.get("shock_unit")

        if delta is None:
            return {
                "success": False,
                "reason": "macro shock value missing",
            }

        if unit != "percentage_point":
            return {
                "success": False,
                "reason": (
                    f"expected percentage_point shock, got {unit}"
                ),
            }

        rate_delta = delta / 100.0

        implied_rate = None

        if interest_expense is not None and total_debt:
            implied_rate = abs(interest_expense) / total_debt

        interest_change = total_debt * rate_delta

        return {
            "success": True,
            "template": template_name,
            "affected_variable": "interest_expense",
            "change_type": "absolute",
            "value": interest_change,
            "unit": "currency_absolute",
            "provenance": {
                "total_debt": total_debt,
                "rate_delta_pp": delta,
                "implied_rate_baseline": implied_rate,
                "formula": "total_debt * (rate_delta_pp / 100)",
            },
        }

    return {
        "success": False,
        "reason": f"unknown template '{template_name}'",
    }


# -------------------------------------------------
# LITERATURE SUB-PIPELINE
# -------------------------------------------------

def literature_company_summary(company_context):

    profile = company_context.get("profile") or {}

    return {
        "company_name": profile.get("company_name"),
        "ticker": profile.get("ticker"),
        "sector": profile.get("sector"),
        "industry": profile.get("industry"),
        "country": profile.get("country"),
    }


def generate_literature_query(mechanism, company_context):

    summary = literature_company_summary(company_context)

    prompt = f"""You design a focused keyword query for OpenAlex.

GOAL
Find ONE empirical quantitative relationship that could be used to
quantify the transmission mechanism below for the company described.

COMPANY
{json.dumps(summary, indent=2)}

TRANSMISSION MECHANISM
{json.dumps(mechanism, indent=2)}

OUTPUT
Return exactly two search strings:

- primary_query: a SHORT keyword query (4-8 content words) naming the
  company's sector or industry and the economic relationship we need.

- fallback_query: an even shorter keyword query (3-5 content words),
  broader but pointing at the same relationship.

CRITICAL CONSTRAINT
OpenAlex performs relevance-based matching but does better with short
keyword queries than with long natural-language questions.

RULES
- Do NOT write questions.
- Do NOT include stopwords.
- Use only content words: nouns, adjectives, verbs, named entities.
- primary_query: 4-8 content words.
- fallback_query: 3-5 content words.
- primary_query must name the company's exact sector or industry.
- fallback_query may name a broader parent category when the exact
  sector is unlikely to have dedicated literature.
- target_quantity must be a concrete measurable quantity.
- quantity_type must be one of:
  "elasticity", "sensitivity", "coefficient", "direction_only".
- Do not invent company-specific numbers.

Return ONLY valid JSON in exactly this shape:

{{
  "primary_query": "...",
  "fallback_query": "...",
  "target_quantity": "...",
  "quantity_type": "..."
}}
"""

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.2,
            response_mime_type="application/json",
        ),
    )

    text = extract_text(response)

    if not text:
        raise ValueError("Gemini returned no literature query.")

    return json.loads(text)


def prefilter_papers(papers):

    seen_dois = set()
    seen_titles = set()
    kept = []

    for paper in papers:

        if not paper.get("abstract"):
            continue

        doi = (paper.get("doi") or "").strip().lower()
        title = (paper.get("title") or "").strip().lower()

        if doi and doi in seen_dois:
            continue

        if title and title in seen_titles:
            continue

        if doi:
            seen_dois.add(doi)

        if title:
            seen_titles.add(title)

        kept.append(paper)

    return kept


def rank_papers_for_mechanism(mechanism, target_quantity, papers):

    if not papers:
        return []

    compact = []

    for paper in papers:
        compact.append({
            "paper_id": paper.get("paper_id"),
            "title": paper.get("title"),
            "year": paper.get("year"),
            "venue": paper.get("venue"),
            "citation_count": paper.get("citation_count"),
            "abstract": paper.get("abstract"),
        })

    prompt = f"""You are ranking candidate papers for a financial stress-test.

MECHANISM
{json.dumps(mechanism, indent=2)}

TARGET QUANTITY WE NEED TO FIND
{target_quantity}

CANDIDATE PAPERS
{json.dumps(compact, indent=2)}

For each candidate paper, decide whether its abstract plausibly contains an
empirical estimate of the target quantity (or a closely related one) that
could be used to quantify this mechanism.

For each paper, output:

- relevance_score: integer 0-10.
    10 = abstract almost certainly contains a directly usable estimate.
     0 = clearly unrelated.

- sector_match: one of:
    "exact"     — the paper studies the same sector or industry as the
                  company, or a strictly narrower one.
    "proxy"     — the paper studies a closely related sector or an
                  aggregate category with a defensible economic analogy
                  to the company.
    "unrelated" — the paper studies a sector with no defensible economic
                  analogy to the company.

- is_relevant: true if and only if
    relevance_score >= 6 AND sector_match is not "unrelated".

- reason: one short sentence explaining your decision.

GUIDANCE
- A "proxy" paper is acceptable if the economic relationship it
  estimates is plausibly transferable to the company's sector.
- Do NOT accept a paper just because it mentions a related keyword.
- Prefer papers whose abstract is explicit about sector, geography,
  period, and the estimated relationship.
- Prefer more recent papers when relevance is otherwise equal.
- Do not use information not present in the supplied data.

Return ONLY valid JSON in exactly this shape:

{{
  "ranked": [
    {{
      "paper_id": "...",
      "relevance_score": 0,
      "sector_match": "exact | proxy | unrelated",
      "is_relevant": false,
      "reason": "one short sentence"
    }}
  ]
}}
"""

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.1,
            response_mime_type="application/json",
        ),
    )

    text = extract_text(response)

    if not text:
        return []

    parsed = json.loads(text)

    scored = {
        entry["paper_id"]: entry
        for entry in parsed.get("ranked", [])
        if entry.get("paper_id")
    }

    by_id = {
        paper["paper_id"]: paper
        for paper in papers
        if paper.get("paper_id")
    }

    relevant = []

    for paper_id, entry in scored.items():
        if not entry.get("is_relevant"):
            continue
        paper = by_id.get(paper_id)
        if paper is None:
            continue
        paper = dict(paper)
        paper["relevance_score"] = entry.get("relevance_score")
        paper["sector_match"] = entry.get("sector_match")
        paper["relevance_reason"] = entry.get("reason")
        relevant.append(paper)

    relevant.sort(
        key=lambda p: p.get("relevance_score") or 0,
        reverse=True
    )

    return relevant[:LITERATURE_TOP_K]


def extract_estimates_from_papers(mechanism, target_quantity, papers):

    if not papers:
        return []

    compact = []

    for paper in papers:
        compact.append({
            "paper_id": paper.get("paper_id"),
            "title": paper.get("title"),
            "year": paper.get("year"),
            "venue": paper.get("venue"),
            "doi": paper.get("doi"),
            "url": paper.get("url"),
            "citation_count": paper.get("citation_count"),
            "sector_match": paper.get("sector_match"),
            "abstract": paper.get("abstract"),
        })

    prompt = f"""You are extracting quantitative estimates from academic abstracts
for a financial stress-test.

MECHANISM
{json.dumps(mechanism, indent=2)}

TARGET QUANTITY
{target_quantity}

PAPERS
{json.dumps(compact, indent=2)}

For every paper, extract every quantitative estimate that could plausibly be
used to quantify the target quantity.

HARD RULES — READ CAREFULLY
1. You may only emit an estimate if the abstract contains an explicit
   sentence stating it. You must include that exact sentence in the
   "quote" field, copied verbatim.
2. If you cannot quote an exact supporting sentence, DO NOT emit the estimate.
3. Do not infer, interpolate or guess a numeric value that the abstract
   does not state.
4. If the abstract reports an effect as statistically insignificant, you may
   emit estimate_value 0.0 with a caveat explaining this. The quote must
   still be present.
5. direction_only is allowed when the abstract states a direction without a
   magnitude; in that case estimate_value must be null and the direction
   field must be set.
6. Do not use prior knowledge about any paper. Only use the fields above.
7. If a paper's sector_match is "proxy", you MUST fill proxy_note to
   explain whether the true value for the company's sector is likely to be
   higher or lower than the paper's estimate, and why. If sector_match is
   "exact", proxy_note must be null.

For each estimate, output:
- paper_id, title, year, venue, doi, url (copied from the paper entry)
- estimate_type: one of
  "elasticity", "sensitivity", "coefficient", "direction_only", "range"
- estimate_value: number or null
- estimate_range: [low, high] or null
- direction: "increase", "decrease", "no_significant_effect", or null
- estimate_unit: short string
- context: population, sector, geography and period
- quote: verbatim snippet from the abstract that states the estimate
- confidence: "high", "medium" or "low"
- caveats: short string or null
- sector_match: "exact" or "proxy" (copied from the paper entry)
- proxy_note: string or null (see hard rule 7)

Return ONLY valid JSON in exactly this shape:

{{
  "estimates": [
    {{
      "paper_id": "...",
      "title": "...",
      "year": 0,
      "venue": "...",
      "doi": "...",
      "url": "...",
      "estimate_type": "...",
      "estimate_value": 0.0,
      "estimate_range": null,
      "direction": null,
      "estimate_unit": "...",
      "context": "...",
      "quote": "...",
      "confidence": "...",
      "caveats": null,
      "sector_match": "...",
      "proxy_note": null
    }}
  ]
}}
"""

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.1,
            response_mime_type="application/json",
        ),
    )

    text = extract_text(response)

    if not text:
        return []

    parsed = json.loads(text)

    estimates = parsed.get("estimates") or []

    return [
        estimate for estimate in estimates
        if estimate.get("quote")
    ]


def quantify_mechanism_via_literature(mechanism, company_context):

    mechanism_label = mechanism.get("mechanism") or "(unnamed mechanism)"

    base = {
        "mechanism": mechanism_label,
        "queries_used": [],
        "target_quantity": None,
        "quantity_type": None,
        "candidates_found": 0,
        "candidates_after_prefilter": 0,
        "shortlist_size": 0,
        "extracted_estimates": [],
    }

    try:
        query_info = generate_literature_query(
            mechanism, company_context
        )
    except Exception as error:
        print(f"[Literature query generation failed: {error}]")
        base["status"] = "query_generation_failed"
        return base

    primary_query = query_info.get("primary_query")
    fallback_query = query_info.get("fallback_query")
    target_quantity = query_info.get("target_quantity")
    quantity_type = query_info.get("quantity_type")

    base["target_quantity"] = target_quantity
    base["quantity_type"] = quantity_type

    if not primary_query:
        base["status"] = "query_generation_failed"
        return base

    ladder = [
        ("primary", primary_query, LITERATURE_RECENT_YEAR),
    ]

    if fallback_query:
        ladder.append(
            ("fallback", fallback_query, LITERATURE_RECENT_YEAR)
        )
        ladder.append(
            ("fallback", fallback_query, LITERATURE_LEGACY_YEAR)
        )

    shortlist = []
    total_candidates = 0

    for label, query, year_from in ladder:

        try:
            results = search_openalex(
                query,
                limit=LITERATURE_SEARCH_LIMIT,
                year_from=year_from,
            )
        except Exception as error:
            print(f"[Literature search failed ({label}): {error}]")
            base["status"] = "search_failed"
            return base

        base["queries_used"].append(
            f"{label}: {query} | from {year_from}"
        )

        candidates = prefilter_papers(results)

        total_candidates += len(candidates)
        base["candidates_found"] = total_candidates
        base["candidates_after_prefilter"] = total_candidates

        if not candidates:
            continue

        try:
            shortlist = rank_papers_for_mechanism(
                mechanism, target_quantity, candidates
            )
        except Exception as error:
            print(f"[Literature ranking failed: {error}]")
            base["status"] = "search_failed"
            return base

        if shortlist:
            break

    base["candidates_found"] = total_candidates
    base["candidates_after_prefilter"] = total_candidates
    base["shortlist_size"] = len(shortlist)

    if not shortlist:
        base["status"] = "no_relevant_papers"
        return base

    try:
        estimates = extract_estimates_from_papers(
            mechanism, target_quantity, shortlist
        )
    except Exception as error:
        print(f"[Literature extraction failed: {error}]")
        base["status"] = "search_failed"
        return base

    base["extracted_estimates"] = estimates
    base["status"] = (
        "extracted" if estimates else "no_extractable_estimate"
    )

    return base


# -------------------------------------------------
# STAGE 3B — EMPIRICAL ROUTE
# -------------------------------------------------

def apply_empirical_route(
    mechanism,
    empirical_target_quantity,
    macro_shock,
    company_context,
):

    lit_mechanism = dict(mechanism)
    lit_mechanism["target_quantity"] = empirical_target_quantity

    lit_result = quantify_mechanism_via_literature(
        lit_mechanism, company_context
    )

    if lit_result.get("status") != "extracted":
        return {
            "success": False,
            "reason": lit_result.get("status"),
            "lit_result": lit_result,
        }

    estimates = lit_result.get("extracted_estimates") or []

    if not estimates:
        return {
            "success": False,
            "reason": "no_estimates",
            "lit_result": lit_result,
        }

    def rank(est):
        sector = est.get("sector_match") or "proxy"
        sector_rank = 0 if sector == "exact" else 1
        confidence = est.get("confidence") or "low"
        confidence_rank = {"high": 0, "medium": 1, "low": 2}.get(
            confidence, 3
        )
        year = est.get("year") or 0
        return (sector_rank, confidence_rank, -year)

    estimates_sorted = sorted(estimates, key=rank)
    chosen = estimates_sorted[0]

    sensitivity = chosen.get("estimate_value")

    if sensitivity is None:
        return {
            "success": False,
            "reason": "chosen_estimate_has_no_value",
            "lit_result": lit_result,
        }

    try:
        sensitivity = float(sensitivity)
    except (TypeError, ValueError):
        return {
            "success": False,
            "reason": "chosen_estimate_not_numeric",
            "lit_result": lit_result,
        }

    shock_unit = macro_shock.get("shock_unit")

    if shock_unit == "percentage_point":
        sensitivity_cap = 0.06
    else:
        sensitivity_cap = 6.0

    cap_hit = False
    if sensitivity > sensitivity_cap:
        sensitivity = sensitivity_cap
        cap_hit = True
    elif sensitivity < -sensitivity_cap:
        sensitivity = -sensitivity_cap
        cap_hit = True

    macro_value = macro_shock.get("shock_value")

    if macro_value is None:
        return {
            "success": False,
            "reason": "macro_shock_missing",
            "lit_result": lit_result,
        }

    company_effect = macro_value * sensitivity

    EFFECT_CAP = 0.60
    if abs(company_effect) > EFFECT_CAP:
        company_effect = EFFECT_CAP * (1 if company_effect > 0 else -1)
        cap_hit = True

    primary_variable = None
    for effect in mechanism.get("financial_effects") or []:
        primary_variable = effect.get("variable")
        break

    alternatives = [
        {
            "estimate_value": e.get("estimate_value"),
            "unit": e.get("estimate_unit"),
            "year": e.get("year"),
            "sector_match": e.get("sector_match"),
            "confidence": e.get("confidence"),
            "title": e.get("title"),
        }
        for e in estimates_sorted[1:]
    ]

    confidence = chosen.get("confidence") or "low"
    if cap_hit:
        confidence = "low"

    return {
        "success": True,
        "affected_variable": primary_variable,
        "change_type": "fractional",
        "value": company_effect,
        "unit": "fractional_change",
        "provenance": {
            "macro_shock_value": macro_value,
            "sensitivity_value": sensitivity,
            "sensitivity_unit": chosen.get("estimate_unit"),
            "paper_id": chosen.get("paper_id"),
            "paper_title": chosen.get("title"),
            "paper_year": chosen.get("year"),
            "paper_doi": chosen.get("doi"),
            "quote": chosen.get("quote"),
            "sector_match": chosen.get("sector_match"),
            "proxy_note": chosen.get("proxy_note"),
            "confidence": confidence,
            "cap_applied": cap_hit,
            "formula": "macro_shock_value * sensitivity_value",
        },
        "alternatives": alternatives,
    }


# -------------------------------------------------
# STAGE 3B — ASSUMPTION ROUTE
# -------------------------------------------------

def apply_assumption_route(
    mechanism,
    macro_shock,
    company_context,
    direction_only_evidence=None,
):

    financial_effects = mechanism.get("financial_effects") or []

    direction = None
    primary_variable = None

    if financial_effects:
        direction = financial_effects[0].get("direction")
        primary_variable = financial_effects[0].get("variable")

    slim_context = {
        "profile": company_context.get("profile"),
        "income_statement": company_context.get("income_statement"),
        "balance_sheet": company_context.get("balance_sheet"),
        "ratios": company_context.get("ratios"),
    }

    shock_unit = macro_shock.get("shock_unit")

    if shock_unit == "percentage_point":
        unit_guidance = (
            "The shock is expressed in PERCENTAGE POINTS.\n"
            "Example: a policy rate shock of +2.48 means the rate\n"
            "rose by 2.48 percentage points (e.g. 4.00% -> 6.48%).\n"
            "The sensitivity is therefore a fractional change in the\n"
            "affected variable PER 1.0 PERCENTAGE POINT of shock.\n"
            "\n"
            "Plausible range: 0.003 to 0.030.\n"
            "\n"
            "Worked examples (shock +2.48pp in all cases):\n"
            "- sensitivity 0.005 -> effect +1.24%\n"
            "- sensitivity 0.02  -> effect +4.96%\n"
            "- sensitivity 0.03  -> effect +7.44%\n"
            "\n"
            "A sensitivity of 0.5 would give a +124% effect from a\n"
            "+2.48pp shock, which is economically absurd."
        )
        sensitivity_cap = 0.06
    else:
        unit_guidance = (
            "The shock is expressed as a FRACTIONAL CHANGE.\n"
            "Example: a GDP shock of -0.0254 means GDP fell by 2.54%.\n"
            "The sensitivity is therefore a fractional change in the\n"
            "affected variable PER 1.0 UNIT of fractional shock.\n"
            "\n"
            "Plausible range: 0.3 to 3.0.\n"
            "\n"
            "Worked examples (shock -0.0254 in all cases):\n"
            "- sensitivity 0.5 -> effect -1.27%\n"
            "- sensitivity 1.4 -> effect -3.56%\n"
            "- sensitivity 3.0 -> effect -7.62%\n"
            "\n"
            "A sensitivity of 30 would give a -76% effect from a\n"
            "-2.54% shock, which is economically absurd."
        )
        sensitivity_cap = 6.0

    prompt = f"""You are the assumption component of a Financial Resilience Agent.

A transmission mechanism could not be quantified from company data
(structural arithmetic) or from the research literature (no quotable
estimate was found). You must propose a sensitivity value, clearly
flagged as an assumption.

MECHANISM
{json.dumps(mechanism, indent=2)}

MACRO SHOCK
{json.dumps(macro_shock, indent=2)}

DIRECTION THE MECHANISM MUST MOVE
{direction}

DIRECTION-ONLY EVIDENCE (may be empty)
{json.dumps(direction_only_evidence or [], indent=2)}

COMPANY CONTEXT
{json.dumps(slim_context, indent=2)}

HOW THE SENSITIVITY IS USED
The pipeline computes:

    company_effect = shock_value * sensitivity_value

where company_effect is a fractional change in the affected variable
(for example, -0.04 means the variable falls by 4%).

{unit_guidance}

Rules:
- Choose a sensitivity value from economic reasoning alone. There is
  no reference library and no external source.
- The sign of the effect must match the DIRECTION stated above.
- If DIRECTION-ONLY EVIDENCE is present, the sign must be consistent
  with it.
- Stay within the plausible range shown above for this shock unit.
- Confidence must be one of "low" or "medium". Use "medium" only when
  the mechanism is a well-understood textbook relationship. Use "low"
  otherwise.
- Justification must be one sentence.

Return ONLY valid JSON in exactly this shape:

{{
  "sensitivity_value": 0.0,
  "justification": "one sentence",
  "confidence": "low"
}}
"""

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.2,
            response_mime_type="application/json",
        ),
    )

    text = extract_text(response)

    if not text:
        return {
            "success": False,
            "reason": "assumption_generation_failed",
        }

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as error:
        return {
            "success": False,
            "reason": f"assumption_json_invalid: {error}",
        }

    sensitivity = parsed.get("sensitivity_value")

    if sensitivity is None:
        return {
            "success": False,
            "reason": "assumption_missing_value",
        }

    try:
        sensitivity = float(sensitivity)
    except (TypeError, ValueError):
        return {
            "success": False,
            "reason": "assumption_value_not_numeric",
        }

    cap_hit = False
    if sensitivity > sensitivity_cap:
        sensitivity = sensitivity_cap
        cap_hit = True
    elif sensitivity < -sensitivity_cap:
        sensitivity = -sensitivity_cap
        cap_hit = True

    macro_value = macro_shock.get("shock_value")

    if macro_value is None:
        return {
            "success": False,
            "reason": "macro_shock_missing",
        }

    company_effect = macro_value * sensitivity

    EFFECT_CAP = 0.60
    if abs(company_effect) > EFFECT_CAP:
        company_effect = EFFECT_CAP * (1 if company_effect > 0 else -1)
        cap_hit = True

    confidence = parsed.get("confidence") or "low"
    if cap_hit:
        confidence = "low"

    return {
        "success": True,
        "affected_variable": primary_variable,
        "change_type": "fractional",
        "value": company_effect,
        "unit": "fractional_change",
        "provenance": {
            "macro_shock_value": macro_value,
            "sensitivity_value": sensitivity,
            "justification": parsed.get("justification"),
            "confidence": confidence,
            "cap_applied": cap_hit,
            "direction_only_evidence_used": bool(
                direction_only_evidence
            ),
            "formula": "macro_shock_value * sensitivity_value",
        },
    }


# -------------------------------------------------
# STAGE 3B — ORCHESTRATOR
# -------------------------------------------------

def run_stage_3b(ticker, company_context, scenario_data):

    scenarios = scenario_data.get("scenarios") or []

    quantified_mechanisms = []
    macro_shocks_used = {}
    warnings = []

    for scenario in scenarios:

        scenario_name = scenario.get("scenario_name") or "(unnamed)"

        driver = scenario.get("macro_driver") or {}
        variable = driver.get("variable")
        direction = driver.get("adverse_direction")

        if not variable or not direction:
            warnings.append(
                f"Scenario '{scenario_name}' has no usable macro driver."
            )
            continue

        macro_shock = get_macro_shock(variable, direction)

        if macro_shock is None:
            warnings.append(
                f"Could not retrieve macro shock for "
                f"'{variable}' ({direction}) in scenario "
                f"'{scenario_name}'."
            )
            continue

        macro_shocks_used[variable] = macro_shock

        if macro_shock["n_observations"] < 200:
            warnings.append(
                f"{variable}: only "
                f"{macro_shock['n_observations']} monthly "
                f"observations available for the 20-year window."
            )

        try:
            classification = classify_scenario_mechanisms(
                scenario, company_context
            )
        except Exception as error:
            warnings.append(
                f"Scenario '{scenario_name}': classification "
                f"failed ({error})."
            )
            continue

        classifications = classification.get("classifications") or []

        mechanisms = scenario.get("transmission_mechanisms") or []

        by_index = {
            entry.get("mechanism_index"): entry
            for entry in classifications
        }

        for idx, mechanism in enumerate(mechanisms):

            entry = by_index.get(idx)

            if entry is None:
                warnings.append(
                    f"Scenario '{scenario_name}', mechanism "
                    f"{idx}: no classification returned."
                )
                continue

            method = entry.get("method")

            if method == "structural":
                route_result = apply_structural_template(
                    entry.get("structural_template"),
                    macro_shock,
                    mechanism,
                    company_context,
                )

            elif method == "empirical":
                route_result = apply_empirical_route(
                    mechanism,
                    entry.get("empirical_target_quantity"),
                    macro_shock,
                    company_context,
                )

                if (
                    not route_result.get("success")
                    and route_result.get("reason") == "search_failed"
                ):
                    print(
                        f"[Retrying empirical route after 5s "
                        f"(scenario='{scenario_name}', "
                        f"mechanism={idx})]"
                    )
                    time.sleep(5)
                    route_result = apply_empirical_route(
                        mechanism,
                        entry.get("empirical_target_quantity"),
                        macro_shock,
                        company_context,
                    )

                if not route_result.get("success"):

                    direction_only_evidence = None

                    lit_result = route_result.get("lit_result") or {}
                    estimates = (
                        lit_result.get("extracted_estimates") or []
                    )
                    direction_only = [
                        e for e in estimates
                        if e.get("estimate_type") == "direction_only"
                    ]
                    if direction_only:
                        direction_only_evidence = direction_only

                    fallback = apply_assumption_route(
                        mechanism,
                        macro_shock,
                        company_context,
                        direction_only_evidence=direction_only_evidence,
                    )

                    if fallback.get("success"):
                        fallback["fell_back_from"] = "empirical"
                        fallback["empirical_failure_reason"] = (
                            route_result.get("reason")
                        )
                        fallback["empirical_was_transient"] = (
                            route_result.get("reason") == "search_failed"
                        )
                        route_result = fallback

                        if route_result.get("empirical_was_transient"):
                            warnings.append(
                                f"Scenario '{scenario_name}', "
                                f"mechanism {idx}: empirical lookup "
                                f"failed twice for a transient reason; "
                                f"fell back to assumption."
                            )

            elif method == "assumption":
                route_result = apply_assumption_route(
                    mechanism,
                    macro_shock,
                    company_context,
                )
            else:
                route_result = {
                    "success": False,
                    "reason": f"unknown_method:{method}",
                }

            quantified_mechanisms.append(
                {
                    "scenario_name": scenario_name,
                    "mechanism_index": idx,
                    "mechanism": mechanism.get("mechanism"),
                    "classification": {
                        "method": method,
                        "reasoning": entry.get("reasoning"),
                        "primary_variable": entry.get(
                            "primary_variable"
                        ),
                    },
                    "macro_driver": {
                        "variable": variable,
                        "adverse_direction": direction,
                    },
                    "macro_shock": macro_shock,
                    "route_result": route_result,
                }
            )

    return {
        "ticker": ticker,
        "quantified_mechanisms": quantified_mechanisms,
        "macro_shocks_used": macro_shocks_used,
        "warnings": warnings,
    }


# -------------------------------------------------
# STAGE 3C — FINANCIAL PROPAGATION
# -------------------------------------------------

def aggregate_scenario_shocks(scenario_mechanisms, company_context):

    income = company_context.get("income_statement") or {}
    cash_flow = company_context.get("cash_flow") or {}

    baseline = {
        "revenue": income.get("revenue"),
        "cogs": None,
        "operating_expenses": None,
        "interest_expense": income.get("interest_expense"),
        "capex": cash_flow.get("capital_expenditure"),
    }

    if baseline["interest_expense"] is not None:
        baseline["interest_expense"] = abs(
            baseline["interest_expense"]
        )

    if (
        baseline["revenue"] is not None
        and income.get("gross_profit") is not None
    ):
        baseline["cogs"] = (
            baseline["revenue"] - income["gross_profit"]
        )

    if (
        income.get("gross_profit") is not None
        and income.get("operating_income") is not None
    ):
        baseline["operating_expenses"] = (
            income["gross_profit"] - income["operating_income"]
        )

    shocks = {
        "revenue": 0.0,
        "cogs": 0.0,
        "operating_expenses": 0.0,
        "interest_expense": 0.0,
        "capex": 0.0,
    }

    applied = []
    unapplied = []

    for mech in scenario_mechanisms:

        route_result = mech.get("route_result") or {}

        if not route_result.get("success"):
            continue

        variable = route_result.get("affected_variable")
        change_type = route_result.get("change_type")
        value = route_result.get("value")

        if variable not in shocks:
            unapplied.append({
                "mechanism_index": mech.get("mechanism_index"),
                "variable": variable,
                "reason": "unsupported_variable",
            })
            continue

        baseline_value = baseline.get(variable)

        if baseline_value is None:
            unapplied.append({
                "mechanism_index": mech.get("mechanism_index"),
                "variable": variable,
                "reason": "baseline_missing",
            })
            continue

        if change_type == "absolute":
            absolute = value
        elif change_type == "fractional":
            absolute = baseline_value * value
        else:
            unapplied.append({
                "mechanism_index": mech.get("mechanism_index"),
                "variable": variable,
                "reason": f"unsupported_change_type:{change_type}",
            })
            continue

        shocks[variable] += absolute

        applied.append({
            "mechanism_index": mech.get("mechanism_index"),
            "variable": variable,
            "change_type": change_type,
            "value": value,
            "absolute_change": absolute,
        })

    return shocks, applied, unapplied, baseline


def propagate_scenario(company_context, shocks, baseline):

    income = company_context.get("income_statement") or {}
    balance = company_context.get("balance_sheet") or {}
    cash_flow = company_context.get("cash_flow") or {}

    revenue = income.get("revenue")
    gross_profit = income.get("gross_profit")
    operating_income = income.get("operating_income")
    ebit = income.get("ebit")
    ebitda = income.get("ebitda")
    net_income = income.get("net_income")
    interest_expense = income.get("interest_expense")

    if interest_expense is not None:
        interest_expense = abs(interest_expense)

    if None in (
        revenue, gross_profit, operating_income,
        net_income, interest_expense,
    ):
        return None

    if revenue == 0:
        return None

    cogs = baseline.get("cogs")
    opex = baseline.get("operating_expenses")

    if cogs is None:
        cogs = revenue - gross_profit

    if opex is None:
        opex = gross_profit - operating_income

    pretax_baseline = (
        (ebit if ebit is not None else operating_income)
        - interest_expense
    )

    if pretax_baseline and pretax_baseline > 0:
        tax_rate = 1.0 - (net_income / pretax_baseline)
        tax_rate = max(0.0, min(0.5, tax_rate))
    else:
        tax_rate = 0.21

    new_revenue = revenue + shocks["revenue"]

    if revenue and revenue > 0:
        cogs_ratio = cogs / revenue
    else:
        cogs_ratio = 0.0

    new_cogs = new_revenue * cogs_ratio + shocks["cogs"]
    new_gross_profit = new_revenue - new_cogs

    new_opex = opex + shocks["operating_expenses"]
    new_operating_income = new_gross_profit - new_opex

    if ebit is not None and operating_income:
        ebit_ratio = ebit / operating_income
    else:
        ebit_ratio = 1.0

    new_ebit = new_operating_income * ebit_ratio

    new_interest = interest_expense + shocks["interest_expense"]

    new_net_income = (new_ebit - new_interest) * (1.0 - tax_rate)

    if ebitda is not None and operating_income is not None:
        new_ebitda = ebitda + (new_operating_income - operating_income)
    else:
        new_ebitda = new_ebit

    baseline_ocf = cash_flow.get("operating_cash_flow")
    baseline_capex = cash_flow.get("capital_expenditure")
    baseline_fcf = cash_flow.get("free_cash_flow")

    if baseline_ocf is not None and net_income is not None:
        new_ocf = baseline_ocf + (new_net_income - net_income)
    else:
        new_ocf = None

    if baseline_capex is not None:
        new_capex = baseline_capex + shocks["capex"]
    else:
        new_capex = None

    if new_ocf is not None and new_capex is not None:
        new_fcf = new_ocf + new_capex
    elif baseline_fcf is not None and net_income is not None:
        new_fcf = baseline_fcf + (new_net_income - net_income)
    else:
        new_fcf = None

    total_debt = balance.get("total_debt")
    baseline_cash = balance.get("cash_and_short_term_investments")
    current_assets = balance.get("current_assets")
    current_liabilities = balance.get("current_liabilities")
    equity = balance.get("stockholders_equity")

    new_equity = None
    if equity is not None and net_income is not None:
        new_equity = equity + (new_net_income - net_income)

    new_cash = None
    if (
        baseline_cash is not None
        and new_fcf is not None
        and baseline_fcf is not None
    ):
        new_cash = baseline_cash + (new_fcf - baseline_fcf)

    new_current_assets = None
    if (
        current_assets is not None
        and new_cash is not None
        and baseline_cash is not None
    ):
        new_current_assets = current_assets + (new_cash - baseline_cash)

    if total_debt is not None and new_cash is not None:
        net_debt = total_debt - new_cash
    elif total_debt is not None and baseline_cash is not None:
        net_debt = total_debt - baseline_cash
    else:
        net_debt = None

    stressed_ratios = {
        "operating_margin": safe_divide(
            new_operating_income, new_revenue
        ),
        "net_profit_margin": safe_divide(
            new_net_income, new_revenue
        ),
        "free_cash_flow_margin": safe_divide(
            new_fcf, new_revenue
        ),
        "current_ratio": safe_divide(
            new_current_assets, current_liabilities
        ),
        "net_debt": net_debt,
        "net_debt_to_ebitda": safe_divide(net_debt, new_ebitda),
        "interest_coverage_ratio": safe_divide(
            new_ebit, new_interest
        ),
    }

    return {
        "stressed_income_statement": {
            "revenue": new_revenue,
            "gross_profit": new_gross_profit,
            "operating_income": new_operating_income,
            "ebit": new_ebit,
            "ebitda": new_ebitda,
            "interest_expense": new_interest,
            "net_income": new_net_income,
        },
        "stressed_balance_sheet": {
            "cash_and_short_term_investments": new_cash,
            "current_assets": new_current_assets,
            "stockholders_equity": new_equity,
        },
        "stressed_cash_flow": {
            "operating_cash_flow": new_ocf,
            "capital_expenditure": new_capex,
            "free_cash_flow": new_fcf,
        },
        "stressed_ratios": stressed_ratios,
    }


def run_stage_3c(company_context, stage_3b_output):

    baseline_ratios = company_context.get("ratios") or {}
    mechanisms = stage_3b_output.get("quantified_mechanisms") or []

    by_scenario = {}

    for mech in mechanisms:
        name = mech.get("scenario_name") or "(unnamed)"
        by_scenario.setdefault(name, []).append(mech)

    stressed_scenarios = []
    warnings = []

    for scenario_name, mechs in by_scenario.items():

        shocks, applied, unapplied, baseline = (
            aggregate_scenario_shocks(mechs, company_context)
        )

        result = propagate_scenario(
            company_context, shocks, baseline
        )

        if result is None:
            warnings.append(
                f"Scenario '{scenario_name}': propagation "
                f"could not run because baseline statements "
                f"are incomplete."
            )
            continue

        driver = None
        macro_shock = None

        if mechs:
            driver = mechs[0].get("macro_driver")
            macro_shock = mechs[0].get("macro_shock")

        stressed_scenarios.append({
            "scenario_name": scenario_name,
            "macro_driver": driver,
            "macro_shock": macro_shock,
            "applied_shocks": applied,
            "unapplied_shocks": unapplied,
            "baseline_ratios": baseline_ratios,
            "stressed_ratios": result["stressed_ratios"],
            "stressed_income_statement":
                result["stressed_income_statement"],
            "stressed_balance_sheet":
                result["stressed_balance_sheet"],
            "stressed_cash_flow":
                result["stressed_cash_flow"],
        })

    return {
        "scenarios": stressed_scenarios,
        "warnings": warnings,
    }


# -------------------------------------------------
# STAGE 4 — RESILIENCE SCORING
# -------------------------------------------------

RATIO_DIRECTIONS = {
    "operating_margin":        "higher_is_better",
    "net_profit_margin":       "higher_is_better",
    "free_cash_flow_margin":   "higher_is_better",
    "current_ratio":           "higher_is_better",
    "net_debt_to_ebitda":      "lower_is_better",
    "interest_coverage_ratio": "higher_is_better",
}

RATIO_THRESHOLDS = {
    "operating_margin":        {"critical": -0.05, "safe": 0.15},
    "net_profit_margin":       {"critical": -0.05, "safe": 0.08},
    "free_cash_flow_margin":   {"critical": -0.05, "safe": 0.08},
    "current_ratio":           {"critical":  0.60, "safe": 1.10},
    "net_debt_to_ebitda":      {"critical":  5.00, "safe": 2.00},
    "interest_coverage_ratio": {"critical":  1.00, "safe": 5.00},
}

PILLARS = {
    "profitability": ["operating_margin", "net_profit_margin"],
    "liquidity":     ["current_ratio"],
    "leverage":      ["net_debt_to_ebitda", "interest_coverage_ratio"],
    "cash_flow":     ["free_cash_flow_margin"],
}


def score_ratio(ratio_name, baseline_value, stressed_value):

    if ratio_name not in RATIO_DIRECTIONS:
        return None

    if baseline_value is None or stressed_value is None:
        return None

    direction = RATIO_DIRECTIONS[ratio_name]
    threshold = RATIO_THRESHOLDS[ratio_name]
    critical = threshold["critical"]
    safe = threshold["safe"]

    if direction == "higher_is_better":

        if baseline_value <= critical:
            relative = 0.0
        else:
            relative = (
                100.0 * (stressed_value - critical)
                / (baseline_value - critical)
            )

        if safe == critical:
            absolute = 0.0
        else:
            absolute = (
                100.0 * (stressed_value - critical)
                / (safe - critical)
            )

        cliff = stressed_value <= critical

    else:

        if baseline_value >= critical:
            relative = 0.0
        else:
            relative = (
                100.0 * (critical - stressed_value)
                / (critical - baseline_value)
            )

        if safe == critical:
            absolute = 0.0
        else:
            absolute = (
                100.0 * (critical - stressed_value)
                / (critical - safe)
            )

        cliff = stressed_value >= critical

    relative = max(0.0, min(100.0, relative))
    absolute = max(0.0, min(100.0, absolute))

    return {
        "ratio": ratio_name,
        "baseline": baseline_value,
        "stressed": stressed_value,
        "relative_score": relative,
        "absolute_score": absolute,
        "ratio_score": min(relative, absolute),
        "cliff_breached": cliff,
    }


def score_scenario(scenario_result):
    """
    Score a single scenario from Stage 3C's output.

    The scenario score is the WEAKEST pillar score, not the mean.
    Resilience is a weakest-link property: a company that collapses
    on one dimension is only as resilient as that dimension, even if
    the others hold up.

    If any ratio breaches its critical threshold, the scenario score
    is capped at CLIFF_CAP (20).
    """

    baseline = scenario_result.get("baseline_ratios") or {}
    stressed = scenario_result.get("stressed_ratios") or {}

    ratio_scores = {}
    cliff_breached_any = False

    for ratio_name in RATIO_DIRECTIONS:

        result = score_ratio(
            ratio_name,
            baseline.get(ratio_name),
            stressed.get(ratio_name),
        )

        if result is None:
            continue

        ratio_scores[ratio_name] = result

        if result["cliff_breached"]:
            cliff_breached_any = True

    pillar_scores = {}

    for pillar, ratios in PILLARS.items():

        vals = [
            ratio_scores[r]["ratio_score"]
            for r in ratios
            if r in ratio_scores
        ]

        if vals:
            pillar_scores[pillar] = sum(vals) / len(vals)

    if not pillar_scores:
        return None

    weakest_pillar_name = min(
        pillar_scores, key=pillar_scores.get
    )
    weakest_pillar_score = pillar_scores[weakest_pillar_name]

    mean_pillar_score = (
        sum(pillar_scores.values()) / len(pillar_scores)
    )

    scenario_score = weakest_pillar_score

    if cliff_breached_any:
        scenario_score = min(scenario_score, 20.0)

    return {
        "scenario_name": scenario_result.get("scenario_name"),
        "ratio_scores": ratio_scores,
        "pillar_scores": pillar_scores,
        "weakest_pillar": weakest_pillar_name,
        "weakest_pillar_score": weakest_pillar_score,
        "mean_pillar_score": mean_pillar_score,
        "cliff_breached": cliff_breached_any,
        "scenario_score": scenario_score,
    }


def classify_score_band(score):
    if score is None:
        return None
    if score >= 80:
        return "High"
    if score >= 60:
        return "Moderate"
    if score >= 40:
        return "Weak"
    return "Critical"


def run_stage_4(stage_3c_output):

    scenario_results = stage_3c_output.get("scenarios") or []

    scored = []

    for scenario in scenario_results:
        result = score_scenario(scenario)
        if result is not None:
            scored.append(result)

    if not scored:
        return {
            "resilience_score": None,
            "min_scenario_score": None,
            "max_scenario_score": None,
            "band": None,
            "scenarios": [],
            "warnings": ["No scenarios could be scored."],
        }

    scores = [s["scenario_score"] for s in scored]

    mean_score = sum(scores) / len(scores)

    return {
        "resilience_score": mean_score,
        "min_scenario_score": min(scores),
        "max_scenario_score": max(scores),
        "band": classify_score_band(mean_score),
        "scenarios": scored,
        "warnings": [],
    }


# -------------------------------------------------
# STAGE 5 — FINAL REPORT
# -------------------------------------------------

def summarize_mechanism_methods(stage_3b_output):
    """
    Count mechanisms by the method that actually produced their value,
    not by their original classification. A mechanism classified as
    empirical but which fell back to assumption is counted as an
    assumption, and the fallback is tallied separately.
    """

    counts = {
        "structural": 0,
        "empirical": 0,
        "assumption": 0,
        "unavailable": 0,
    }
    empirical_fell_back = 0
    low_confidence = []
    capped = []

    for mech in stage_3b_output.get("quantified_mechanisms") or []:

        classification = mech.get("classification") or {}
        route_result = mech.get("route_result") or {}
        provenance = route_result.get("provenance") or {}

        classified_method = classification.get("method")
        fell_back = route_result.get("fell_back_from") == "empirical"
        success = bool(route_result.get("success"))

        if not success:
            actual_method = "unavailable"
        elif fell_back:
            actual_method = "assumption"
            empirical_fell_back += 1
        elif classified_method in counts:
            actual_method = classified_method
        else:
            actual_method = "unavailable"

        counts[actual_method] += 1

        if provenance.get("confidence") == "low":
            low_confidence.append({
                "scenario": mech.get("scenario_name"),
                "mechanism": mech.get("mechanism"),
                "actual_method": actual_method,
                "justification": provenance.get("justification"),
            })

        if provenance.get("cap_applied"):
            capped.append({
                "scenario": mech.get("scenario_name"),
                "mechanism": mech.get("mechanism"),
                "actual_method": actual_method,
                "sensitivity_value": provenance.get("sensitivity_value"),
            })

    return {
        "counts": counts,
        "empirical_fell_back_to_assumption": empirical_fell_back,
        "low_confidence_assumptions": low_confidence,
        "capped_mechanisms": capped,
    }


def run_stage_5(
    ticker,
    baseline_assessment,
    stage_3b_output,
    stage_3c_output,
    stage_4_output,
):

    method_summary = summarize_mechanism_methods(stage_3b_output)
    def _fmt_score_obj(obj):
        """
        Recursively format numeric values in a dict/list to strings
        with 3 significant figures. Returns a new structure; the
        original is not modified.
        """

        if isinstance(obj, dict):
            return {k: _fmt_score_obj(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_fmt_score_obj(v) for v in obj]
        if isinstance(obj, bool):
            return obj
        if isinstance(obj, (int, float)):
            return to_significant_figures(obj)

        return obj

    stage_4_display = _fmt_score_obj(stage_4_output)
    confidence_data = {
        "method_counts": method_summary["counts"],
        "empirical_fell_back_to_assumption":
            method_summary["empirical_fell_back_to_assumption"],
        "low_confidence_assumptions":
            method_summary["low_confidence_assumptions"],
        "capped_mechanisms":
            method_summary["capped_mechanisms"],
        "pipeline_warnings": list(
            (stage_3b_output.get("warnings") or [])
            + (stage_3c_output.get("warnings") or [])
            + (stage_4_output.get("warnings") or [])
        ),
    }

    prompt = f"""You are the final report writer of a Financial Resilience Agent.

Company ticker: {ticker}

Use ONLY the data supplied below.
Do not invent any numbers.
Do not use prior knowledge about the company.
Do not recalculate ratios.

BASELINE ASSESSMENT (from Stage 2)
{baseline_assessment}

STAGE 4 RESILIENCE SCORE
{json.dumps(stage_4_display, indent=2)}

STAGE 3C STRESSED SCENARIOS
{json.dumps(stage_3c_output, indent=2)}

STAGE 3B MECHANISM SUMMARY
{json.dumps(confidence_data, indent=2)}

Write the report in markdown with EXACTLY these five sections:

## 1. Baseline Context
Two or three sentences on the company, its sector, and its baseline
financial position. Draw on the baseline assessment above.

## 2. Overall Resilience Score
State the mean resilience score, its band, and the min and max
scenario scores. Explain that the scenario score is the weakest
pillar, so the score reflects the tightest constraint under stress.

## 3. Score Composition
For each scenario, state:
- its resilience score
- the macro driver and its 95th-percentile adverse shock value
- the four pillar sub-scores (profitability, liquidity, leverage,
  cash flow)
- the weakest pillar for that scenario
Present this as a markdown table, one row per scenario.

## 4. Scenario Detail
For each scenario, one or two paragraphs covering:
- what the scenario is
- the mechanisms that mattered most
- how the key ratios changed from baseline to stressed
State fiscal year only. Margins as percentages. Ratios as multiples.

## 5. Confidence and Limitations
State how many mechanisms actually used each quantification method
(structural / empirical / assumption / unavailable). If any empirical
mechanisms fell back to assumption, say so explicitly with the count.
Flag any low-confidence or capped mechanisms. List any pipeline
warnings. End with a one-line verdict on how well-supported the score
is.

Rules:
- Margins as percentages with one decimal place.
- Ratios as multiples with two decimal places.
- Scores, shock values, and any other numeric values from the
  Stage 4 data are already formatted to 3 significant figures in
  the JSON above. Reproduce them exactly as they appear — do not
  re-format, round, or expand them.
- For fractional-change shocks, include the percentage equivalent
  in parentheses.
- Do not add sections or subsections beyond the five above.
- Do not include a preamble or a closing summary outside section 5.
"""

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.2
        ),
    )

    return extract_text(response)


def print_report(report_text):
    print("\n")
    print("=" * 70)
    print("FINAL REPORT")
    print("=" * 70)
    print(report_text)
    print("=" * 70)


# -------------------------------------------------
# MAIN PROGRAM
# -------------------------------------------------

def main():

    print("\nFinancial Resilience Agent")
    print("--------------------------")

    ticker = input(
        "\nEnter a company ticker: "
    )

    ticker = ticker.strip().upper()

    if not ticker:
        print("Please enter a ticker.")
        return

    get_company_data.cache_clear()
    fetch_fred_series_cached.cache_clear()
    get_macro_shock.cache_clear()
    search_openalex.cache_clear()

    print(f"\nAgent analysing {ticker}...\n")

    try:

        print("[Stage 1] Retrieving company data...")

        profile = get_company_profile(ticker)
        income_statement = get_income_statement(ticker)
        balance_sheet = get_balance_sheet(ticker)
        cash_flow = get_cash_flow(ticker)

        ratios = calculate_baseline_ratios(
            income_statement,
            balance_sheet,
            cash_flow
        )

        company_context = {
            "profile": profile,
            "income_statement": income_statement,
            "balance_sheet": balance_sheet,
            "cash_flow": cash_flow,
            "ratios": ratios,
        }

        print("[Stage 2] Generating baseline assessment...")

        baseline_assessment = run_financial_agent(
            ticker,
            company_context
        )

        print("[Stage 3A] Generating stress scenarios...")

        scenario_data = run_scenario_agent(
            ticker,
            company_context
        )

        print("[Stage 3B] Quantifying transmission mechanisms...")

        stage_3b_output = run_stage_3b(
            ticker,
            company_context,
            scenario_data
        )

        print("[Stage 3C] Propagating shocks through statements...")

        stage_3c_output = run_stage_3c(
            company_context,
            stage_3b_output
        )

        print("[Stage 4] Computing resilience scores...")

        stage_4_output = run_stage_4(
            stage_3c_output
        )

        print("[Stage 5] Writing final report...")

        report = run_stage_5(
            ticker,
            baseline_assessment,
            stage_3b_output,
            stage_3c_output,
            stage_4_output,
        )

        print_report(report)

    except Exception as error:

        print("\nSomething went wrong.")
        print(error)


if __name__ == "__main__":
    main()