"""
Inspired and Adopted Codes from:
    https://github.com/multimodal-ai-lab/DEFAME/tree/main
Please check and cite
"""

from urllib.parse import urlparse
import re
from PIL import Image
import os
from bs4 import BeautifulSoup
import datetime

from markdownify import MarkdownConverter
MEDIA_REF_REGEX = r"(<(?:image|video|audio):[0-9]+>)"
MEDIA_ID_REGEX = r"(?:<(?:image|video|audio):([0-9]+)>)"
MEDIA_SPECIFIER_REGEX = r"(?:<(image|video|audio):([0-9]+)>)"

URL_REGEX = r"https?:\/\/(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9@:%_\+.~#?&//=]*)"

ROOT_DIR=os.path.abspath('../../..')
IMAGE_FOLDER=os.path.join(ROOT_DIR,'download_img_from_urls')
ERROR_LOGGER_FOLDER=os.path.join(ROOT_DIR,'error_logger')

import errno
import os

import re
from typing import Optional

import numpy as np
from sty import fg, Style, RgbFg

fg.orange = Style(RgbFg(255, 150, 50))


def gray(text: str):
    return fg.da_grey + text + fg.rs


def light_blue(text: str):
    return fg.li_blue + text + fg.rs


def green(text: str):
    return fg.green + text + fg.rs


def yellow(text: str):
    return fg.li_yellow + text + fg.rs


def red(text: str):
    return fg.red + text + fg.rs


def magenta(text: str):
    return fg.magenta + text + fg.rs


def cyan(text: str):
    return fg.cyan + text + fg.rs


def orange(text: str):
    return fg.orange + text + fg.rs


def bold(text):  # boldface
    return "\033[1m" + text + "\033[0m"


def it(text):  # italic
    return "\033[3m" + text + "\033[0m"


def ul(text):  # underline
    return "\033[4m" + text + "\033[0m"


class Scrap_Logger(object):
    def __init__(self,output_dir):
        dirname=os.path.dirname(output_dir)
        if not os.path.exists(dirname):
            os.mkdir(dirname)
        self.log_file=open(output_dir,'w')
        self.infos={}
        self.color_mapper={
            'red':red,
            'yellow':yellow,
            'gray':gray,
            'light_blue':light_blue,
            'orange':orange,
            'green':green,
            'magenta':magenta,
            'cyan':cyan
        }
        
    def append(self,key,val):
        vals=self.infos.setdefault(key,[])
        vals.append(val)

    def log(self,extra_msg=''):
        msgs=[extra_msg]
        for key, vals in self.infos.iteritems():
            msgs.append('%s %.6f' %(key,np.mean(vals)))
        msg='\n'.joint(msgs)
        self.log_file.write(msg+'\n')
        self.log_file.flush()
        self.infos={}
        return msg
        
    def write(self,color,msg):
        msg = msg.encode("utf-8", "ignore").decode("utf-8")
        if color=='black':
            self.log_file.write(msg+'\n')
        else:
            self.log_file.write(self.color_mapper[color](msg)+'\n')
        self.log_file.flush()
        print(msg)    

logger = Scrap_Logger(os.path.join(ERROR_LOGGER_FOLDER,'web_scrap_logger.txt'))

block_keywords = [
        "captcha",
        "verify you are human",
        "access denied",
        "premium content",
        "403 Forbidden",
        "You have been blocked",
        "Please enable JavaScript",
        "I'm not a robot",
        "Are you a robot?",
        "Are you a human?",
    ]

fact_checking_urls = [
    "snopes.com",
    "politifact.com",
    "factcheck.org",
    "truthorfiction.com",
    "fullfact.org",
    "leadstories.com",
    "hoax-slayer.net",
    "checkyourfact.com",
    "reuters.com/fact-check",
    "reuters.com/article/fact-check",
    "apnews.com/APFactCheck",
    "factcheck.afp.com",
    "poynter.org",
    "factcheck.ge",
    "vishvasnews.com",
    "boomlive.in",
    "altnews.in",
    "thequint.com/news/webqoof",
    "factcheck.kz",
    "data.gesis.org/claimskg/claim_review",
]

unscrapable_urls = [
    "https://www.thelugarcenter.org/ourwork-Bipartisan-Index.html",
    "https://data.news-leader.com/gas-price/",
    "https://www.wlbt.com/2023/03/13/3-years-later-mississippis-/",
    "https://edition.cnn.com/2021/01/11/business/no-fl",
    "https://www.thelugarcenter.org/ourwork-Bipart",
    "https://www.linkedin.com/pulse/senator-kelly-/",
    "http://libres.uncg.edu/ir/list-etd.aspx?styp=ty&bs=master%27s%20thesis&amt=100000",
    "https://www.washingtonpost.com/investigations/coronavirus-testing-denials/2020/03/",

]

unsupported_domains = [
    "facebook.com",
    "twitter.com",
    "x.com",
    "instagram.com",
    "youtube.com",
    "tiktok.com",
    "reddit.com",
    "ebay.com",
    "microsoft.com",
    "researchhub.com",
    "pinterest.com",
    "irs.gov"
]

firecrawl_url = "http://firecrawl:3002"  # applies to Firecrawl running in a 'firecrawl' Docker Container

import requests
def _firecrawl_is_running():
    """Returns True iff Firecrawl is running."""
    try:
        response = requests.get(firecrawl_url)
    except (requests.exceptions.ConnectionError, requests.exceptions.RetryError):
        return False
    return response.status_code == 200

def get_domain(url):
    parsed_url = urlparse(url)
    netloc = parsed_url.netloc.lower()  # get the network location (netloc)
    domain = '.'.join(netloc.split('.')[-2:])  # remove subdomains
    return domain


def is_fact_checking_site(url):
    """Check if the URL belongs to a known fact-checking website."""
    # Check if the domain matches any known fact-checking website
    for site in fact_checking_urls:
        if site in url:
            return True
    return False

def is_unsupported_site(url):
    """
    Checks if the URL belongs to a known unsupported website.
    """
    if (".gov" in url) or (url in unscrapable_urls):
        return True
    domain = get_domain(url)
    return domain in unsupported_domains

def get_markdown_hyperlinks(text):
    """Extracts all web hyperlinks from the given markdown-formatted string. Returns
    a list of hypertext-URL-pairs."""
    hyperlink_regex = f'(?:\\[([^]^[]*)\\]\\(({URL_REGEX})\\))'
    pattern = re.compile(hyperlink_regex, re.DOTALL)
    hyperlinks = re.findall(pattern, text)
    return hyperlinks

def is_image_url(url):
    """Returns True iff the URL points at an accessible _pixel_ image file."""
    try:
        response = requests.head(url, timeout=2)
        content_type = response.headers.get('content-type')
        return (content_type.startswith("image/") and not "svg" in response.headers.get('content-type') and
                not "svg" in content_type and
                not "eps" in content_type)
    except Exception:
        return False

def _resolve_media_hyperlinks(text):
    """Identifies up to MAX_MEDIA_PER_PAGE image URLs, downloads the images and replaces the
    respective Markdown hyperlinks with their proper image reference."""
    if text is None:
        return None
    hyperlinks = get_markdown_hyperlinks(text)
    media_count = 0
    for hypertext, url in hyperlinks:
        # Check if URL is an image URL
        if is_image_url(url) and not is_fact_checking_site(url) and not is_unsupported_site(url):
            try:
                # TODO: Handle very large images like: https://eoimages.gsfc.nasa.gov/images/imagerecords/144000/144225/campfire_oli_2018312_lrg.jpg
                # Download the image
                response = requests.get(url, stream=True, timeout=10)
                if response.status_code == 200:
                    img = Image.open(io.BytesIO(response.content)).convert('RGB')
                    path_to_file = os.path.join(IMAGE_FOLDER,datetime.now().strftime("%Y-%m-%d_%H-%M-%s-%f") + ".jpg")
                    logger.write("black",'\tHaving image!! and saving at %s, the %d image' % (path_to_file,media_count))
                    img.save(path_to_file)
                    # Replace the Markdown hyperlink; Interleaved image text
                    text = text.replace(f"[{hypertext}]({url})", f"{hypertext} {path_to_file}")
                    media_count += 1
                    if media_count >= MAX_MEDIA_PER_PAGE:
                        break
                    else:
                        continue

            except (requests.exceptions.ConnectTimeout, requests.exceptions.ConnectionError, requests.exceptions.ReadTimeout, requests.exceptions.TooManyRedirects):
                # Webserver is not reachable (anymore)
                pass

            except UnidentifiedImageError as e:
                print(f"Unable to download image from {url}.")
                print(e)
                # Image has an incompatible format. Skip it.
                pass

            except Exception as e:
                print(f"Unable to download image from {url}.")
                print(e)
                pass

            finally:
                # Remove the hyperlink, just keep the hypertext
                text = text.replace(f"[{hypertext}]({url})", "")

        # TODO: Resolve videos and audios
    #return MultimediaSnippet(text)
    return text

def log_error_url(url, message):
    error_log_file = os.path.join(ERROR_LOGGER_FOLDER,"crawl_error_log.txt")
    with open(error_log_file, "a") as f:
        f.write(f"{url}: {message}\n")

def scrape_firecrawl(url):
    """Scrapes the given URL using Firecrawl. Returns a Markdown-formatted
    multimedia snippet, containing any (relevant) media from the page."""
    headers = {
        'Content-Type': 'application/json',
    }
    json_data = {
        "url": url,
        "formats": ["markdown"],
        "timeout": 15 * 60 * 1000,  # waiting time in milliseconds for Firecrawl to process the job
    }

    try:
        response = requests.post(firecrawl_url + "/v1/scrape",
                                 json=json_data,
                                 headers=headers,
                                 timeout=10 * 60)  # Firecrawl scrapes usually take 2 to 4s, but a 1700-page PDF takes 5 min
    except (requests.exceptions.RetryError, requests.exceptions.ConnectionError):
        loggerwrite(red, "Firecrawl is not running!")
        return None
    except requests.exceptions.Timeout:
        error_message = "Firecrawl failed to respond in time! This can be due to server overload."
        logger.write("orange",f"{error_message}\nSkipping the URL {url}.")
        log_error_url(url, error_message)
        return None
    except Exception as e:
        error_message = f"Exception: {repr(e)}"
        logger.write("yellow",repr(e))
        logger.write("yellow", f"Unable to scrape {url} with Firecrawl. Skipping...")
        log_error_url(url, error_message)
        return None

    if response.status_code != 200:
        logger.write("black",f"Failed to scrape {url}")
        error_message = f"Failed to scrape {url} - Status code: {response.status_code} - Reason: {response.reason}"
        log_error_url(url, error_message)
        if response.status_code==402:
            logger.write("black",f"Error 402: Access denied.")
        elif response.status_code==403:
            logger.write("black",f"Error 403: Forbidden.")
        elif response.status_code==408: 
            logger.write("orange"f"Error 408: Timeout! Firecrawl overloaded or Webpage did not respond.")
        elif response.status_code==409:
            logger.write("black",f"Error 409: Access denied.")
        elif response.status_code==500: 
            logger.write("black",f"Error 500: Server error.")
        else:
            logger.write("black",f"Error {response.status_code}: {response.reason}.")
        logger.write("black","Skipping that URL.")
        return None

    success = response.json()["success"]
    if success and "data" in response.json():
        data = response.json()["data"]
        text = data.get("markdown")
        return _resolve_media_hyperlinks(text)
    else:
        error_message = f"Unable to read {url}. No usable data in response."
        logger.write("black",f"Unable to read {url}. Skipping it.")
        logger.write("black",str(response.content))
        log_error_url(url, error_message)
        return None

def postprocess_scraped(text):
    # Remove any excess whitespaces
    text = re.sub(r' {2,}', ' ', text)
    # remove any excess newlines
    text = re.sub(r'(\n *){3,}', '\n\n', text)
    return text

def scrape_naive(url):
    """Fallback scraping script."""
    # TODO: Also scrape images
    headers = {
        'User-Agent': 'Mozilla/4.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    }
    try:
        page = requests.get(url, headers=headers, timeout=5)

        # Handle any request errors
        if page.status_code == 403:
            #logger.write("black",f"Forbidden URL: {url}")
            return None
        elif page.status_code == 404:
            return None
        page.raise_for_status()

        soup = BeautifulSoup(page.content, 'html.parser')
        # text = soup.get_text(separator='\n', strip=True)
        if soup.article:
            # News articles often use the <article> tag to mark article contents
            soup = soup.article
        # Turn soup object into a Markdown-formatted string
        #text = md(soup)
        text=MarkdownConverter().convert_soup(soup)
        text = postprocess_scraped(text)
        #text=_resolve_media_hyperlinks(text)
        return text
    except requests.exceptions.Timeout:
        logger.write("black",f"Timeout occurred while naively scraping {url}")
    except requests.exceptions.HTTPError as http_err:
        logger.write("black",f"HTTP error occurred while doing naive scrape: {http_err}")
    except requests.exceptions.RequestException as req_err:
        logger.write("black",f"Request exception occurred while scraping {url}: {req_err}")
    except Exception as e:
        logger.write("black",f"An unexpected error occurred while scraping {url}: {e}")
    return None

def is_relevant_content(content):
    """Checks if the web scraping result contains relevant content or is blocked by a bot-catcher/paywall."""
    if not content:
        return False
    # Check for suspiciously short content (less than 500 characters might indicate blocking)
    if len(content.strip()) < 500:
        return False
    for keyword in block_keywords:
        if re.search(keyword, content, re.IGNORECASE):
            return False
    return True

def scrape(url):
    """
    Scrapes the contents of the specified webpage.
    Code adopted from: https://github.com/multimodal-ai-lab/DEFAME/blob/main/defame/tools/search/remote_search_api.py#L157
    """
    if is_unsupported_site(url):
        #logger.write("black",f"Skipping unsupported site {url}.")
        return None
    """
    if _firecrawl_is_running():
        scraped = scrape_firecrawl(url)
        
    else:
        logger.write("orange",f"Firecrawl is not running! Falling back to Beautiful Soup.")
        scraped = scrape_naive(url)
    """
    #cur set naive as default
    scraped = scrape_naive(url)
    if scraped and is_relevant_content(str(scraped)):
        return scraped
    else:
        return None


if __name__ == '__main__':
    logger.write("black","Running scrapes with Firecrawl...")
    urls_to_scrape = [
        #"https://www.washingtonpost.com/video/national/cruz-calls-trump-clinton-two-new-york-liberals/2016/04/07/da3b78a8-fcdf-11e5-813a-90ab563f0dde_video.html",
        "https://nypost.com/2024/10/11/us-news/meteorologists-hit-with-death-threats-after-debunking-hurricane-conspiracy-theories/",
        #"https://www.tagesschau.de/ausland/asien/libanon-israel-blauhelme-nahost-102.html",
        #"https://www.zeit.de/politik/ausland/2024-10/wolodymyr-selenskyj-berlin-olaf-scholz-militaerhilfe",
        #"https://edition.cnn.com/2024/10/07/business/property-damange-hurricane-helene-47-billion/index.html"
    ]
    for url in urls_to_scrape:
        scraped = scrape(url)
        if scraped:
            print(scraped, "\n\n\n")
        else:
            logger.write("red","Scrape failed.")