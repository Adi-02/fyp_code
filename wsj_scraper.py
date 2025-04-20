from firecrawl import FirecrawlApp
import os 
from dotenv import load_dotenv
import re

load_dotenv()
app = FirecrawlApp(api_key=os.getenv("firecrawlkey"))

def get_general_news(start):
    start_date = start.replace("-", "/")
    print(f"Scraping data for: {start_date}")
    
    relevant_categories = {
        "Business", "Tech", "Economy", "Markets", "U.S. Markets", "Autos Industry", 
        "Finance", "Banking", "Real Estate", "Central Banks", "Politics", 
        "Risk & Compliance Journal", "Energy", "Commodities", "Retail", 
        "Review & Outlook", "Commentary", "World"
    }

    try:
        url = f"https://www.wsj.com/news/archive/{start_date}"
        response = app.scrape_url(url=url, params={'formats': ['markdown']})
        markdown_data = response.get("markdown", "")

        if markdown_data.strip() == "":
            print(f"No markdown data found for {start_date}")
            return []

        lines = markdown_data.splitlines()
        current_category = None
        relevant_titles = []

        for line in lines:
            line = line.strip()

            if line and not line.startswith(("#", "-", "[", "!", "Page", "Skip")) and not re.search(r"\d{1,2}:\d{2} (AM|PM) ET", line):
                current_category = line

            match = re.match(r"## \[([^\]]+)\]", line)
            if match and current_category in relevant_categories:
                title = match.group(1)
                relevant_titles.append(title)

        return relevant_titles

    except Exception as e:
        print(f"Error on {start_date}: {e}")
        return []

if __name__ == "__main__":
    titles = get_general_news("2025-03-03")
    print(titles)
