import os, json, time, datetime, sqlite3
from typing import List, Literal
from dotenv import load_dotenv
from pydantic import BaseModel
import pandas as pd
from groq import Groq
from tavily import TavilyClient

load_dotenv()
llm = Groq(api_key=os.environ["GROQ_API_KEY"])
search_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
MODEL = "openai/gpt-oss-120b"


# ---------- LLM helper (JSON mode + retries for rate limits) ----------
def ask_llm(prompt, schema=None):
    messages = [{"role": "user", "content": prompt}]
    kwargs = dict(model=MODEL, temperature=0.2)
    if schema:
        messages = [
            {"role": "system", "content":
                "Respond ONLY with valid JSON that matches this JSON schema:\n"
                + json.dumps(schema.model_json_schema())},
            {"role": "user", "content": prompt},
        ]
        kwargs["response_format"] = {"type": "json_object"}
    for attempt in range(4):
        try:
            r = llm.chat.completions.create(messages=messages, **kwargs)
            text = r.choices[0].message.content
            return schema.model_validate_json(text) if schema else text
        except Exception as e:
            print("LLM error, retrying:", e)
            if attempt == 3:
                raise
            time.sleep(6 * (attempt + 1))


# ---------- Schemas ----------
class Plan(BaseModel):
    intent: Literal["competitor_tracking", "benchmarking", "sentiment", "whitespace", "general"]
    company: str
    competitors: List[str]
    queries: List[str]

class Gaps(BaseModel):
    enough: bool
    followup_queries: List[str]

class Attr(BaseModel):
    name: str
    value: str

class Product(BaseModel):
    brand: str
    product: str
    price: str
    attributes: List[Attr]
    source_ids: List[int]

class Benchmark(BaseModel):
    products: List[Product]

class Fact(BaseModel):
    brand: str
    category: Literal["pricing", "feature", "positioning", "customer_complaint",
                      "customer_praise", "launch_or_news", "market_data", "other"]
    fact: str

class Facts(BaseModel):
    facts: List[Fact]


# ---------- Storage (for competitor tracking) ----------
db = sqlite3.connect("market.db", check_same_thread=False)
db.execute("CREATE TABLE IF NOT EXISTS seen(url TEXT PRIMARY KEY, title TEXT, first_seen TEXT)")

def only_new(sources):
    fresh = []
    for s in sources:
        if db.execute("SELECT 1 FROM seen WHERE url=?", (s["url"],)).fetchone() is None:
            db.execute("INSERT INTO seen VALUES (?,?,?)",
                       (s["url"], s["title"], datetime.datetime.now().isoformat()))
            fresh.append(s)
    db.commit()
    return fresh


# ---------- Tools ----------
def web_search(q, news=False, n=5):
    kw = dict(query=q, max_results=n, search_depth="advanced", include_raw_content=True)
    if news:
        kw.update(topic="news", days=30)
    res = search_client.search(**kw)
    return [{
        "title": r["title"],
        "url": r["url"],
        "content": r["content"],
        "raw": (r.get("raw_content") or r["content"])[:6000],
    } for r in res["results"]]

def gather(queries, news=False, sources=None):
    sources = sources or []
    seen = {s["url"] for s in sources}
    for q in queries:
        try:
            results = web_search(q, news=news)
        except Exception as e:
            print("search failed:", q, e)
            continue
        for r in results:
            if r["url"] in seen:
                continue
            seen.add(r["url"])
            r["id"] = len(sources) + 1
            sources.append(r)
    return sources

def fmt(sources, limit=500):
    return "\n\n".join(
        f"[{s['id']}] {s['title']} ({s['url']})\n{s['content'][:limit]}" for s in sources
    )

def fmt_facts(facts):
    return "\n".join(
        f"[{f['source_id']}] ({f['brand']} | {f['category']}) {f['fact']}" for f in facts
    )


# ---------- Agent steps ----------
def make_plan(question, context=""):
    return ask_llm(f"""You are a market research planner. Today is {datetime.date.today()}.
Business context: {context or 'not provided'}
User question: {question}

Choose the intent:
- competitor_tracking: recent moves, pricing changes, launches, messaging
- benchmarking: compare products/features/prices
- sentiment: what consumers think, complaints, trends
- whitespace: unmet needs, gaps, opportunities
- general: anything else

Name the company and up to 4 real, direct competitors (infer them if not given).
Write 8-10 specific search queries. Rules:
- At least one query per competitor about pricing or plans
- At least one query per competitor about reviews and complaints (add words like 'reddit', 'review', 'complaints', 'vs')
- Include the country or market from the context in queries when relevant
- Use specific product names, never vague phrases like 'market trends'""", Plan)

def reflect(question, sources):
    return ask_llm(f"""Question: {question}
Evidence so far:
{fmt(sources, 400)[:12000]}

Is this enough to answer well? If not, give up to 3 follow-up search queries
targeting what's missing.""", Gaps)

def extract_facts(sources, company, log=print):
    all_facts = []
    for s in sources:
        try:
            out = ask_llm(f"""Extract concrete facts from this page that matter for market research on {company} and its competitors.
Keep only specifics: prices, plan names, limits, features, dates, named complaints or praise, market numbers.
Skip generic marketing language. Each fact must be one short sentence that stands alone.
Return an empty list if the page has nothing useful.

Source: {s['title']}
{s['raw'][:3000]}""", Facts)
            for f in out.facts:
                all_facts.append({**f.model_dump(), "source_id": s["id"]})
        except Exception as e:
            log(f"Skipped source {s['id']}: {e}")
        time.sleep(2)  # keeps you under free-tier rate limits
    return all_facts

def benchmark_table(facts):
    data = ask_llm(f"""From these facts, build a product comparison. For each product give brand, product,
price, and attributes (plan limits, features, positioning, audience). Use only the facts given.
Use 'not found' for unknown values.

{fmt_facts(facts)[:12000]}""", Benchmark)
    rows = {}
    for p in data.products:
        col = f"{p.brand} - {p.product}"
        rows[col] = {"Price": p.price}
        for a in p.attributes:
            rows[col][a.name] = a.value
    return pd.DataFrame(rows).fillna("-")

INSTRUCTIONS = {
    "competitor_tracking": "List each new competitor move (pricing, launches, campaigns, messaging changes) with date if known, and explain why it matters.",
    "benchmarking": "Compare the products across the attributes found. Highlight where the company leads, lags, or is at parity.",
    "sentiment": "Summarize consumer sentiment per brand: positive themes, negative themes, and any shift in recent months. Quote specific complaints or praise.",
    "whitespace": "Identify unmet consumer needs that no competitor serves well, by contrasting customer complaints with competitor offerings. Rank by opportunity size and justify each with evidence.",
    "general": "Answer the question thoroughly.",
}

def analyze(question, plan, facts, context=""):
    return ask_llm(f"""You are a senior market analyst writing for a decision-maker.
Use ONLY the extracted facts below. Cite each claim with its source id like [3].
Never use outside knowledge. If a topic has no facts, say "no evidence found" instead of guessing.
Prefer specific numbers, plan names and quotes over general statements.
Avoid generic advice like "focus on customer experience".

Business context: {context or 'not provided'}
Question: {question}
Company: {plan.company} | Competitors: {', '.join(plan.competitors)}
Task: {INSTRUCTIONS[plan.intent]}

Facts:
{fmt_facts(facts)[:14000]}

Write markdown with these sections:
1. Executive summary (3 bullets, each with a concrete finding)
2. Evidence table (brand | what we found | source id)
3. Competitor comparison (where the company leads, lags, or matches)
4. Opportunities and risks (each tied to specific evidence)
5. Recommended actions (specific, ranked, with reasoning)
6. Evidence gaps (what we could not verify, and what data would fix it)""")


# ---------- Orchestrator ----------
def run(question, context="", log=print):
    log("Planning...")
    plan = make_plan(question, context)
    log(f"Intent: {plan.intent} | Company: {plan.company} | Competitors: {', '.join(plan.competitors)}")

    news = plan.intent == "competitor_tracking"
    log(f"Searching ({len(plan.queries)} queries)...")
    sources = gather(plan.queries, news=news)

    log("Checking for gaps...")
    g = reflect(question, sources)
    if not g.enough and g.followup_queries:
        log(f"Follow-up searches: {g.followup_queries}")
        sources = gather(g.followup_queries[:3], news=news, sources=sources)

    if plan.intent == "competitor_tracking":
        sources = only_new(sources)
        if not sources:
            return {"report": "No new competitor activity since the last run.",
                    "table": None, "sources": [], "facts": []}

    sources = sources[:12]                      # cap to control cost and rate limits
    for i, s in enumerate(sources, 1):          # clean citation numbers
        s["id"] = i

    log(f"Reading {len(sources)} pages and extracting facts (takes a minute)...")
    facts = extract_facts(sources, plan.company, log)
    log(f"Extracted {len(facts)} facts")

    if not facts:
        return {"report": "No usable facts were extracted. Try a more specific question or add business context.",
                "table": None, "sources": sources, "facts": []}

    table = None
    if plan.intent == "benchmarking":
        log("Building comparison table...")
        try:
            table = benchmark_table(facts)
        except Exception as e:
            log(f"Table extraction failed: {e}")

    log("Writing report...")
    report = analyze(question, plan, facts, context)
    return {"report": report, "table": table, "sources": sources, "facts": facts}