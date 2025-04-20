import asyncio
import json
import os
import re
import time
from datetime import datetime
import pandas as pd
from dotenv import load_dotenv
from pathlib import Path
from playwright.async_api import async_playwright

load_dotenv()

def delay_setting(min_delay=0.5, max_delay=2.0):
    time.sleep(min_delay + (max_delay - min_delay) * os.urandom(1)[0] / 255)

async def scrape_seeking_alpha(ticker: str, date: str):
    url = f"https://seekingalpha.com/symbol/{ticker}/news?from={date}&to={date}"
    print(f"Scraping {ticker} on {date} from {url}")

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context()
        page = await context.new_page()

        await page.goto(url)

        # Simulate human-like behavior
        await page.mouse.move(100, 100)
        await asyncio.sleep(0.5)
        await page.mouse.move(200, 300)
        await asyncio.sleep(0.5)
        await page.mouse.move(300, 400)
        await asyncio.sleep(0.5)
        await page.mouse.move(400, 200)
        await asyncio.sleep(0.5)

        # Simulate scrolling
        for _ in range(3):
            await page.mouse.wheel(0, 300)
            await asyncio.sleep(1)
            await page.mouse.move(100 + _*50, 200 + _*30)

        await asyncio.sleep(2)

        content = await page.content()
        titles = re.findall(r"<h3[^>]*>(.*?)</h3>", content)
        clean_titles = [re.sub(r"<.*?>", "", title).strip() for title in titles if title.strip()]

        await browser.close()

        return clean_titles if clean_titles else None

async def main():
    start_date = "2025-03-28"
    end_date = "2025-03-31"

    business_days = pd.bdate_range(start=start_date, end=end_date)
    tickers = ["AAPL", "AMZN", "BA", "DIS", "F", "GOOG", "KO", "MO", "MSFT", "T"]

    for ticker in tickers:
        print("Processing:", ticker)
        filename = f"financial_data/{ticker}_news.json"

        if os.path.exists(filename):
            with open(filename, "r") as f:
                ticker_data = json.load(f)
        else:
            ticker_data = {}

        Path("financial_data").mkdir(exist_ok=True)

        i = 0
        for single_day in business_days:
            if i == 5:
                print("Taking longer delay to avoid rate-limiting...")
                time.sleep(10)
                i = 0

            date_str = single_day.strftime("%Y-%m-%d")

            if date_str in ticker_data:
                i += 1
                continue

            titles = await scrape_seeking_alpha(ticker, date_str)

            if titles:
                formatted_titles = [{"title": title} for title in titles]
                ticker_data[date_str] = formatted_titles

                with open(filename, "w") as f:
                    json.dump(ticker_data, f, indent=4)

                print(f"Added news for {ticker} on {date_str}")

            i += 1
            delay_setting(5, 10)

if __name__ == "__main__":
    asyncio.run(main())
