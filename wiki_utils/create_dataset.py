import requests

def extract_articles_from_list(page_title):
    """Extract article links with page IDs from a Wikipedia list page"""
    try:
        url = "https://en.wikipedia.org/w/api.php"
        headers = {'User-Agent': 'WikipediaBot/1.0 (Educational Project)'}
        
        all_articles = []
        continue_params = {}
        
        while True:
            params = {
                'action': 'query',
                'generator': 'links',
                'titles': page_title,
                'gpllimit': 'max',
                'gplnamespace': 0,
                'format': 'json'
            }
            params.update(continue_params)
            
            response = requests.get(url, params=params, headers=headers, timeout=10)
            data = response.json()
            
            if 'query' in data and 'pages' in data['query']:
                articles = [(page.get('title'), page.get('pageid')) 
                           for page in data['query']['pages'].values()
                           if page.get('pageid') is not None]
                all_articles.extend(articles)
            
            # Check for continuation
            if 'continue' in data:
                continue_params = data['continue']
            else:
                break
        
        return all_articles
        
    except Exception as e:
        print(f"  Error extracting from {page_title}: {e}")
        return []