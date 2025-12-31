#!/usr/bin/env python3
"""
FOMC Calendar Fetcher - Free alternatives to get FOMC meeting data

Integrated into AutoTrade v5.2 for event-driven predictions.

Usage:
    from src.ml.fomc_calendar import get_fomc_from_fed_website, get_next_fomc_meeting

    # Get all FOMC dates
    data = get_fomc_from_fed_website()

    # Get next meeting
    next_meeting = get_next_fomc_meeting()
    # {'next_meeting': '2025-12-10', 'days_until': 3}
"""

import requests
from datetime import datetime, date
from bs4 import BeautifulSoup
import json


def get_fomc_from_fed_website():
    """
    Scrape FOMC calendar directly from Federal Reserve website
    Most reliable source - official data
    """
    url = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"

    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
    }

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')

        meetings = []

        # Find all meeting panels
        panels = soup.find_all('div', class_='fomc-meeting')

        for panel in panels:
            meeting = {}

            # Get month/year
            month_elem = panel.find('div', class_='fomc-meeting__month')
            if month_elem:
                meeting['month'] = month_elem.get_text(strip=True)

            # Get dates
            date_elem = panel.find('div', class_='fomc-meeting__date')
            if date_elem:
                meeting['dates'] = date_elem.get_text(strip=True)

            # Check for statement/projection
            if panel.find('a', string=lambda t: t and 'Statement' in t):
                meeting['has_statement'] = True

            if panel.find(string=lambda t: t and 'projection' in t.lower()):
                meeting['has_projections'] = True

            if meeting:
                meetings.append(meeting)

        return {'source': 'Federal Reserve', 'meetings': meetings}

    except Exception as e:
        return {'error': str(e)}


def get_fomc_from_investing_com():
    """
    Scrape economic calendar from Investing.com
    Filter for FOMC events
    """
    url = "https://www.investing.com/economic-calendar/"

    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
    }

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')

        events = []

        # Find calendar table rows
        rows = soup.find_all('tr', class_='js-event-item')

        for row in rows:
            event_name = row.get('data-event', '')

            # Filter for FOMC/Fed related events
            fomc_keywords = ['FOMC', 'Federal Reserve', 'Fed Interest Rate',
                           'Fed Chair', 'Powell', 'Fed Funds Rate']

            if any(keyword.lower() in event_name.lower() for keyword in fomc_keywords):
                event = {
                    'name': event_name,
                    'id': row.get('data-event-id'),
                }

                # Get time
                time_cell = row.find('td', class_='time')
                if time_cell:
                    event['time'] = time_cell.get_text(strip=True)

                # Get currency/country
                currency_cell = row.find('td', class_='flagCur')
                if currency_cell:
                    event['currency'] = currency_cell.get_text(strip=True)

                # Get impact (bulls icons)
                impact_cell = row.find('td', class_='sentiment')
                if impact_cell:
                    bulls = impact_cell.find_all('i', class_='grayFullBullishIcon')
                    event['impact'] = len(bulls)

                events.append(event)

        return {'source': 'Investing.com', 'events': events}

    except Exception as e:
        return {'error': str(e)}


def get_hardcoded_fomc_2025():
    """
    Hardcoded FOMC dates for 2025 - from official Fed schedule
    Most reliable, no scraping needed
    """
    meetings = [
        {'dates': 'January 28-29', 'month': 'January', 'year': 2025, 'has_projections': False},
        {'dates': 'March 18-19', 'month': 'March', 'year': 2025, 'has_projections': True},
        {'dates': 'May 6-7', 'month': 'May', 'year': 2025, 'has_projections': False},
        {'dates': 'June 17-18', 'month': 'June', 'year': 2025, 'has_projections': True},
        {'dates': 'July 29-30', 'month': 'July', 'year': 2025, 'has_projections': False},
        {'dates': 'September 16-17', 'month': 'September', 'year': 2025, 'has_projections': True},
        {'dates': 'October 28-29', 'month': 'October', 'year': 2025, 'has_projections': False},
        {'dates': 'December 9-10', 'month': 'December', 'year': 2025, 'has_projections': True},
    ]

    return {'source': 'Hardcoded (Official Fed Schedule)', 'year': 2025, 'meetings': meetings}


def get_next_fomc_meeting():
    """
    Get the next upcoming FOMC meeting date
    """
    meetings_2025 = [
        date(2025, 1, 29),
        date(2025, 3, 19),
        date(2025, 5, 7),
        date(2025, 6, 18),
        date(2025, 7, 30),
        date(2025, 9, 17),
        date(2025, 10, 29),
        date(2025, 12, 10),
    ]

    today = date.today()

    for meeting_date in meetings_2025:
        if meeting_date >= today:
            days_until = (meeting_date - today).days
            return {
                'next_meeting': meeting_date.isoformat(),
                'days_until': days_until,
                'formatted': meeting_date.strftime('%B %d, %Y')
            }

    return {'message': 'No more FOMC meetings in 2025'}


def get_all_fomc_dates_2025() -> list:
    """
    Get all FOMC dates as date objects for integration with LiveEventFetcher.

    Returns:
        List of datetime.date objects for all 2025 FOMC meetings
    """
    return [
        date(2025, 1, 29),
        date(2025, 3, 19),
        date(2025, 5, 7),
        date(2025, 6, 18),
        date(2025, 7, 30),
        date(2025, 9, 17),
        date(2025, 10, 29),
        date(2025, 12, 10),
    ]


if __name__ == '__main__':
    print("=" * 60)
    print("FOMC CALENDAR TEST")
    print("=" * 60)

    # Test 1: Hardcoded dates (always works)
    print("\n[TEST 1] Hardcoded FOMC 2025 Schedule:")
    print("-" * 40)
    result = get_hardcoded_fomc_2025()
    print(json.dumps(result, indent=2))

    # Test 2: Next meeting
    print("\n[TEST 2] Next FOMC Meeting:")
    print("-" * 40)
    result = get_next_fomc_meeting()
    print(json.dumps(result, indent=2))

    # Test 3: Scrape Fed website
    print("\n[TEST 3] Scraping Federal Reserve Website:")
    print("-" * 40)
    result = get_fomc_from_fed_website()
    if 'error' in result:
        print(f"Error: {result['error']}")
    else:
        print(f"Found {len(result.get('meetings', []))} meetings")
        print(json.dumps(result, indent=2))

    # Test 4: Scrape Investing.com
    print("\n[TEST 4] Scraping Investing.com Calendar:")
    print("-" * 40)
    result = get_fomc_from_investing_com()
    if 'error' in result:
        print(f"Error: {result['error']}")
    else:
        print(f"Found {len(result.get('events', []))} FOMC events")
        print(json.dumps(result, indent=2))
