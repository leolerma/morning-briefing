import feedparser
import yfinance as yf
from openai import OpenAI
from datetime import datetime, timezone
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# -------------------------
# CONFIG
# -------------------------
RSS_FEEDS = {
    "EV Market": "https://news.google.com/rss/search?q=electric+vehicle+market",
    "Semiconductors": "https://news.google.com/rss/search?q=semiconductor+industry",
    "Schaeffler": "https://news.google.com/rss/search?q=Schaeffler",
    "Lear Corporation": "https://news.google.com/rss/search?q=Lear+Corporation"
}

STOCKS = {
    "Lear Corporation": "LEA",
    "onsemi" : "ON",
    "Schaeffler" : "SFFLY"
}

OUTPUT_DIR = "docs/audio"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------
# DATA FETCH
# -------------------------
def fetch_news():
    articles = []
    for topic, url in RSS_FEEDS.items():
        feed = feedparser.parse(url)
        for entry in feed.entries[:4]:
            articles.append(f"{topic}: {entry.title}")
    return "\n".join(articles)

def fetch_stock_summary():
    summaries = []
    for name, ticker in STOCKS.items():
        stock = yf.Ticker(ticker)
        hist = stock.history(period="5d")
        if len(hist) >= 2:
            last = hist["Close"].iloc[-1]
            prev = hist["Close"].iloc[-2]
            change = last - prev
            pct = (change / prev) * 100
            summaries.append(
                f"{name} ({ticker}) closed at ${last:.2f}, "
                f"change {change:+.2f} ({pct:+.2f}%)"
            )
    return "\n".join(summaries)

# -------------------------
# SCRIPT GENERATION
# -------------------------
def generate_script(news, stocks):
    today = datetime.now(timezone.utc).strftime("%A, %B %d")
    prompt = f"""
You are a professional financial news podcast host. Your name is Batman.

Create a 6–8 minute morning audio briefing for {today}.

Structure:
1. Short intro
2. EV market and auto market overview
3. Semiconductor industry overview, with particular attention to companies like onsemi, Texas Instruments, Infenion, Wolfspeed, MPS, and STmicro
4. Schaeffler update + why it matters
5. Lear Corporation update + why it matters
6. Stock performance summary of onsemi, Schaeffler, and Lear
7. Upcoming catalysts (earnings, guidance, industry events)
8. Short outlook
9. Clean outro

Please only pull recent news from the past week for everything, nothing older than a week. Be specific in latest news for the semiconductor companies and Schaeffler and Lear. use recent annoncements, and latest news. If there is nothing relavent, then it is ok to not mention anything.

Tone: calm, analytical, confident
Audience: engineer / automotive / semiconductor sales professional

News:
{news}

Stock Data:
{stocks}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# -------------------------
# TEXT TO SPEECH
# -------------------------
def text_to_speech(text, filename):
    with client.audio.speech.with_streaming_response.create(
        model="gpt-4o-mini-tts",
        voice="alloy",
        input=text
    ) as response:
        response.stream_to_file(filename)

# -------------------------
# MAIN
# -------------------------
if __name__ == "__main__":
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    audio_path = f"{OUTPUT_DIR}/briefing_{date_str}.mp3"

    print("Fetching news...")
    news = fetch_news()

    print("Fetching stock data...")
    stocks = fetch_stock_summary()

    print("Generating script...")
    script = generate_script(news, stocks)

    print("Converting to speech (this may take ~1 minute)...")
    text_to_speech(script, audio_path)

    print("Done! Morning briefing generated:", audio_path)
