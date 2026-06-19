# search.py
import os
import asyncio
import random
import re
from playwright.async_api import async_playwright
from dotenv import load_dotenv

load_dotenv()

GOOGLE_BASE_URL = "https://www.google.com/search"
SEARCH_LANGUAGE = os.getenv("SEARCH_LANG", "en-US")
SEARCH_REGION = os.getenv("SEARCH_REGION", "US")
PROXY_SERVER = None  # 强制禁用代理，如需启用请设置代理地址

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
]

# 尝试加载 playwright-stealth
STEALTH_AVAILABLE = False
try:
    from playwright_stealth import stealth_async
    STEALTH_AVAILABLE = True
except ImportError:
    pass

async def google_ai_search(query: str, retry: int = 1) -> str:
    for attempt in range(retry + 1):
        async with async_playwright() as p:
            browser = None
            try:
                launch_args = {
                    "headless": False,
                    "args": [
                        '--disable-blink-features=AutomationControlled',
                        '--no-sandbox',
                        '--disable-setuid-sandbox',
                        '--disable-dev-shm-usage',
                        '--disable-web-security',
                    ],
                    "ignore_default_args": ["--enable-automation"]
                }
                if PROXY_SERVER:
                    launch_args["proxy"] = {"server": PROXY_SERVER}
                
                browser = await p.chromium.launch(**launch_args)
                context = await browser.new_context(
                    user_agent=random.choice(USER_AGENTS),
                    viewport={"width": random.choice([1366, 1440, 1536]), "height": random.choice([768, 900, 864])},
                    locale=SEARCH_LANGUAGE,
                    timezone_id="America/New_York",
                )
                await context.add_init_script("""
                    Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
                    window.navigator.chrome = { runtime: {} };
                    Object.defineProperty(navigator, 'plugins', { get: () => [1,2,3,4,5] });
                    Object.defineProperty(navigator, 'languages', { get: () => ['en-US', 'en'] });
                """)
                
                page = await context.new_page()
                
                if STEALTH_AVAILABLE:
                    await stealth_async(page)
                
                await page.evaluate("window.scrollTo(0, Math.random() * 100);")
                await asyncio.sleep(random.uniform(1, 2))
                
                search_url = f"{GOOGLE_BASE_URL}?q={query}&hl={SEARCH_LANGUAGE}&gl={SEARCH_REGION}&udm=50"
                await page.goto(search_url, wait_until="domcontentloaded", timeout=200000)
                
                await asyncio.sleep(random.uniform(3, 6))
                await page.evaluate("window.scrollBy(0, Math.random() * 300);")
                await asyncio.sleep(random.uniform(2, 4))
                
                body_text = await page.inner_text("body")
                if "unusual traffic" in body_text or "captcha" in body_text.lower():
                    if attempt == retry:
                        return "❌ Google 检测到异常流量，请更换 IP 或稍后再试。"
                    else:
                        await browser.close()
                        await asyncio.sleep(random.uniform(10, 15))
                        continue
                
                ai_text = "未找到 AI 生成的直接回答。"
                rso = await page.query_selector("#rso")
                if rso:
                    ai_div = await rso.query_selector("[data-attrid='kc']") or rso
                    full_text = await ai_div.inner_text()
                    ai_text = full_text.strip()[:1500]
                    if len(full_text) > 1500:
                        ai_text += "..."
                else:
                    ai_text = body_text[:1500].strip()
                
                sources = []
                links = await page.evaluate("""() => {
                    const anchors = Array.from(document.querySelectorAll('a'));
                    return anchors
                        .map(a => a.href)
                        .filter(href => href.startsWith('http') && !href.includes('google.com') && !href.includes('maps.google'));
                }""")
                sources = list(dict.fromkeys(links))[:10]
                if not sources:
                    urls = re.findall(r'https?://[^\s\)\]>]+', body_text)
                    sources = [url for url in urls if 'google.com' not in url]
                    sources = list(dict.fromkeys(sources))[:10]
                
                sources_text = "\n".join([f"[{i+1}] {url}" for i, url in enumerate(sources)]) if sources else "No external sources found."
                
                await browser.close()
                # 关闭浏览器后随机等待 5~15 秒，降低请求频率
                await asyncio.sleep(random.uniform(5, 10))
                return f"""
--- Google AI Search Reply ---
{ai_text}

--- Sources ---
{sources_text}
"""
                
            except Exception as e:
                if browser:
                    await browser.close()
                if attempt == retry:
                    return f"❌ 搜索执行错误: {str(e)}"
                else:
                    await asyncio.sleep(random.uniform(5, 10))
                    continue
    return "❌ 多次重试后失败。"

async def close_browser():
    pass