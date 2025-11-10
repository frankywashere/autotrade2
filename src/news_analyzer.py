"""News analyzer with AI sentiment and BS scoring."""
import requests
import json
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from dataclasses import dataclass
import anthropic
import config


@dataclass
class NewsArticle:
    """Data class for news article."""
    title: str
    description: str
    url: str
    published_at: datetime
    source: str
    sentiment: Optional[str] = None
    sentiment_score: Optional[float] = None
    bs_score: Optional[float] = None
    analysis: Optional[str] = None


class NewsAnalyzer:
    """Analyze news with AI-powered sentiment and BS detection."""

    def __init__(self, stock: str = config.DEFAULT_STOCK):
        """
        Initialize news analyzer.

        Args:
            stock: Stock symbol to analyze news for
        """
        self.stock = stock
        self.client = anthropic.Anthropic(api_key=config.CLAUDE_API_KEY)
        self.news_cache: List[NewsArticle] = []
        self.last_fetch: Optional[datetime] = None

    def fetch_news(self, hours_back: int = 24) -> List[NewsArticle]:
        """
        Fetch news articles for the stock.

        Args:
            hours_back: Number of hours to look back

        Returns:
            List of NewsArticle objects
        """
        articles = []

        # Try NewsAPI if key is available
        if config.NEWS_API_KEY:
            articles.extend(self._fetch_from_newsapi(hours_back))

        # If no NewsAPI or no results, use fallback sources
        if not articles:
            articles.extend(self._fetch_fallback_news(hours_back))

        self.news_cache = articles
        self.last_fetch = datetime.now()

        return articles

    def _fetch_from_newsapi(self, hours_back: int) -> List[NewsArticle]:
        """Fetch from NewsAPI."""
        try:
            from_date = (datetime.now() - timedelta(hours=hours_back)).isoformat()
            url = f"https://newsapi.org/v2/everything"
            params = {
                'q': f"{self.stock} OR Tesla" if self.stock == "TSLA" else self.stock,
                'from': from_date,
                'sortBy': 'publishedAt',
                'apiKey': config.NEWS_API_KEY,
                'language': 'en'
            }

            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            articles = []
            for item in data.get('articles', [])[:20]:  # Limit to 20 articles
                articles.append(NewsArticle(
                    title=item.get('title', ''),
                    description=item.get('description', ''),
                    url=item.get('url', ''),
                    published_at=datetime.fromisoformat(item.get('publishedAt', '').replace('Z', '+00:00')),
                    source=item.get('source', {}).get('name', 'Unknown')
                ))

            return articles

        except Exception as e:
            print(f"Error fetching from NewsAPI: {e}")
            return []

    def _fetch_fallback_news(self, hours_back: int) -> List[NewsArticle]:
        """
        Fallback method to create sample news articles.
        In production, this would scrape from other sources.
        """
        # Return mock articles for testing
        now = datetime.now()
        return [
            NewsArticle(
                title=f"{self.stock} Stock Shows Volatility Amid Market Uncertainty",
                description=f"Recent trading shows {self.stock} experiencing increased volatility as investors react to market conditions.",
                url="",
                published_at=now - timedelta(hours=2),
                source="Mock News"
            ),
            NewsArticle(
                title=f"Analysts Remain Divided on {self.stock} Future Prospects",
                description=f"Wall Street analysts are split on their outlook for {self.stock} in the coming quarter.",
                url="",
                published_at=now - timedelta(hours=5),
                source="Mock News"
            )
        ]

    def analyze_article(self, article: NewsArticle, stock_context: Optional[str] = None) -> NewsArticle:
        """
        Analyze article sentiment and BS score using Claude AI.

        Args:
            article: NewsArticle to analyze
            stock_context: Optional context about current stock conditions

        Returns:
            Updated NewsArticle with analysis
        """
        try:
            # Build prompt for Claude
            prompt = f"""Analyze this news article about {self.stock}:

Title: {article.title}
Description: {article.description}
Published: {article.published_at}

{f"Current Market Context: {stock_context}" if stock_context else ""}

Please provide:
1. Sentiment: positive, negative, or neutral
2. Sentiment Score: -100 (very bearish) to +100 (very bullish)
3. BS Score: 0-100 (0 = factual, 100 = likely BS/clickbait/overreaction)
   - Consider: Is this rehashed old news? Clickbait headline? Contradicts historical patterns?
   - Historical context: If similar bearish news previously led to rebounds, score as high BS
   - If article seems designed to create panic/FOMO without substance, increase BS score
4. Brief analysis (1-2 sentences)

Format your response as JSON:
{{
  "sentiment": "positive/negative/neutral",
  "sentiment_score": <number>,
  "bs_score": <number>,
  "analysis": "<brief explanation>"
}}"""

            # Call Claude API
            message = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=500,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )

            # Parse response
            response_text = message.content[0].text

            # Extract JSON from response (handle markdown code blocks)
            if "```json" in response_text:
                json_str = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                json_str = response_text.split("```")[1].split("```")[0].strip()
            else:
                json_str = response_text.strip()

            result = json.loads(json_str)

            # Update article
            article.sentiment = result.get('sentiment', 'neutral')
            article.sentiment_score = float(result.get('sentiment_score', 0))
            article.bs_score = float(result.get('bs_score', 50))
            article.analysis = result.get('analysis', '')

        except Exception as e:
            print(f"Error analyzing article: {e}")
            # Set defaults on error
            article.sentiment = "neutral"
            article.sentiment_score = 0
            article.bs_score = 50
            article.analysis = f"Error during analysis: {str(e)}"

        return article

    def analyze_all_news(self, stock_context: Optional[str] = None) -> List[NewsArticle]:
        """
        Analyze all cached news articles.

        Args:
            stock_context: Optional context about current stock conditions

        Returns:
            List of analyzed NewsArticle objects
        """
        if not self.news_cache:
            self.fetch_news()

        analyzed_articles = []
        for article in self.news_cache:
            if article.sentiment is None:  # Not yet analyzed
                analyzed_article = self.analyze_article(article, stock_context)
                analyzed_articles.append(analyzed_article)
            else:
                analyzed_articles.append(article)

        return analyzed_articles

    def get_overall_sentiment(self) -> Dict:
        """
        Get overall news sentiment summary.

        Returns:
            Dictionary with aggregated sentiment data
        """
        if not self.news_cache:
            return {
                "avg_sentiment_score": 0,
                "avg_bs_score": 50,
                "positive_count": 0,
                "negative_count": 0,
                "neutral_count": 0,
                "high_bs_count": 0,
                "signal": "neutral"
            }

        analyzed = [a for a in self.news_cache if a.sentiment is not None]

        if not analyzed:
            return {
                "avg_sentiment_score": 0,
                "avg_bs_score": 50,
                "positive_count": 0,
                "negative_count": 0,
                "neutral_count": 0,
                "high_bs_count": 0,
                "signal": "neutral"
            }

        # Calculate averages
        avg_sentiment = sum(a.sentiment_score for a in analyzed) / len(analyzed)
        avg_bs = sum(a.bs_score for a in analyzed) / len(analyzed)

        # Count sentiments
        positive = sum(1 for a in analyzed if a.sentiment == "positive")
        negative = sum(1 for a in analyzed if a.sentiment == "negative")
        neutral = sum(1 for a in analyzed if a.sentiment == "neutral")
        high_bs = sum(1 for a in analyzed if a.bs_score > config.BS_SCORE_THRESHOLD)

        # Determine signal
        if avg_bs > config.BS_SCORE_THRESHOLD:
            signal = "ignore"  # High BS, ignore the news
        elif avg_sentiment > 20:
            signal = "positive"
        elif avg_sentiment < -20:
            signal = "negative"
        else:
            signal = "neutral"

        return {
            "avg_sentiment_score": avg_sentiment,
            "avg_bs_score": avg_bs,
            "positive_count": positive,
            "negative_count": negative,
            "neutral_count": neutral,
            "high_bs_count": high_bs,
            "total_articles": len(analyzed),
            "signal": signal,
            "recommendation": self._get_recommendation(signal, avg_bs, avg_sentiment)
        }

    def _get_recommendation(self, signal: str, bs_score: float, sentiment_score: float) -> str:
        """Generate trading recommendation based on news."""
        if signal == "ignore":
            return "Ignore bearish news (high BS score) - potential buy the dip opportunity"
        elif signal == "negative" and bs_score < 40:
            return "Genuine negative news - exercise caution"
        elif signal == "positive" and bs_score < 40:
            return "Positive news with low BS - bullish signal"
        else:
            return "Mixed signals - defer to technical analysis"


if __name__ == "__main__":
    # Test news analyzer
    analyzer = NewsAnalyzer("TSLA")

    print("Fetching news...")
    articles = analyzer.fetch_news(hours_back=48)
    print(f"Found {len(articles)} articles\n")

    print("Analyzing articles with Claude AI...")
    stock_context = "TSLA is trading near lower channel line with oversold RSI"
    analyzed = analyzer.analyze_all_news(stock_context)

    print("\n" + "=" * 80)
    for i, article in enumerate(analyzed[:5], 1):  # Show first 5
        print(f"\nArticle {i}:")
        print(f"Title: {article.title}")
        print(f"Source: {article.source}")
        print(f"Sentiment: {article.sentiment} (score: {article.sentiment_score:+.1f})")
        print(f"BS Score: {article.bs_score:.1f}/100")
        print(f"Analysis: {article.analysis}")
        print("-" * 80)

    # Overall sentiment
    overall = analyzer.get_overall_sentiment()
    print("\n" + "=" * 80)
    print("OVERALL NEWS SENTIMENT:")
    print(f"Average Sentiment Score: {overall['avg_sentiment_score']:+.1f}")
    print(f"Average BS Score: {overall['avg_bs_score']:.1f}")
    print(f"Positive: {overall['positive_count']}, Negative: {overall['negative_count']}, Neutral: {overall['neutral_count']}")
    print(f"High BS Articles: {overall['high_bs_count']}/{overall['total_articles']}")
    print(f"\nSignal: {overall['signal']}")
    print(f"Recommendation: {overall['recommendation']}")
