import requests
from bs4 import BeautifulSoup

# Send a request to the webpage
url = 'https://blog.govtribe.com/top-20-federal-contracting-opportunities-in-january-2024'
response = requests.get(url)

# Check if the request was successful
if response.status_code == 200:
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Get all text from the page
    page_content = soup.get_text()

    # Save the scraped content to a text file
    with open('scraped_content.txt', 'w', encoding='utf-8') as file:
        file.write(page_content)
    
    print('Content has been saved to "scraped_content.txt".')
else:
    print('Failed to retrieve the webpage')
