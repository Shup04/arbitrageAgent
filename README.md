# arbitrageAgent

A local Facebook Marketplace scouting experiment: find newly listed tools, batteries, bikes and repair candidates, then use a vision model to decide which listings deserve a closer look for resale.

The implemented workflow is a single Python script. It collects listings through Apify, keeps a SQLite memory of previously seen items, and asks OpenAI to identify and appraise each candidate. Its output is a console report, not an automated buying system. Scraping and appraisal are connected, but reliable appraisal and repeat-run processing are unfinished.

## The idea behind the project

The search terms in [main.py](main.py#L24) make the intended niche unusually concrete: `18650`, `surron`, `broken`, `cnc`, `3d printer` and `makita battery`. Together with a local-city filter, they point toward finding useful hardware and repairable items close enough to collect. The code supports that interpretation; it does not establish the author's repair expertise or any history of profitable purchases.

The difficult part is recognizing what a cheap listing actually offers. A low price for a Sur-Ron could mean an underpriced bike, a damaged bike, or a footpeg. The [appraisal prompt](main.py#L154) explicitly uses the bike-versus-footpeg example and asks the model to infer condition, estimate conservative resale value in CAD, and consider asking price plus repair effort. Photographs are therefore part of the reasoning input, rather than just decoration for a search result.

There is a second idea around controlling the cost of that search: scrape broadly, avoid paying to appraise the same listing twice, and eventually adjust collection volume to recent activity. The seen-item database implements the first part. The seven-day volume calculation exists as a helper, but it is not connected to the active scraper.

This is best understood as an experiment in reducing the manual effort of finding candidates. A model's `is_deal` answer is a hypothesis for human review, not an observed resale profit.

## How it works

```text
Search terms + Marketplace area
             |
             v
Apify Marketplace scraper
             |
             v
Allowed-city filter -> SQLite seen-item check
             |
             v
Title + price + matched keyword + optional listing image
             |
             v
OpenAI structured appraisal
             |
             v
Console explanation + in-memory winning_deals list
```

1. **Configuration and clients.** `load_dotenv()` loads credentials; both API keys are required before the script can proceed. Search terms, allowed cities and the Marketplace search-area identifier are embedded in the source. The current cities are Kamloops, Chase, Merritt, Darfield, Kelowna and Vernon.
2. **Collection.** [`run_scraper()`](main.py#L59) builds one search URL per term, requesting newest-first results. It calls `crawlerbros/facebook-marketplace-scraper` with `resultsLimit=150`, Canadian residential proxies and `includeListingDetails=False`. These are the actor settings in this revision, not a guarantee of current actor compatibility.
3. **Local filtering and memory.** A location can arrive as text or an object; the code handles both and checks for an allowed city name. It derives an item ID from the URL, excludes IDs already in SQLite, and keeps the title, price, URL, primary image URL and first matching search keyword in memory.
4. **Appraisal.** [`evaluate_deal()`](main.py#L137) attempts to download the primary image with a five-second timeout. If successful, it sends a base64 image alongside the text prompt; otherwise it continues with text alone. The configured model is `gpt-4o-mini`.
5. **Structured results.** The requested JSON schema contains `identified_item`, `is_accessory_or_part`, `estimated_resale_value`, `is_deal` and `reasoning`. This gives the script fields it can consume, but does not validate the appraisal itself.
6. **Review output.** The [main loop](main.py#L214) skips empty or zero prices, prints successful appraisals and collects positive results in `winning_deals`. It prints a final count and exits. It does not save the winning list, send notifications, message sellers or buy anything.

## Files and persistent state

| File or object | Purpose |
| --- | --- |
| [`main.py`](main.py) | Configuration, scraping, database helpers, image retrieval, appraisal and entry point. |
| `.env` | Optional local credential file, excluded by `.gitignore`. |
| `arbitrage_memory.db` | Created in the working directory; excluded by `.gitignore`. |
| `seen_listings` table | Stores item ID, title and date first recorded. It does not store appraisal results or a processing status. |
| `volume_history` table | Stores date/keyword/new-count entries for the unused adaptive-volume helpers. |
| `winning_deals` | Python list containing listing details and positive appraisals; lost when the process exits. |

## Where development stopped

The two recorded commits are from April 8, 2026. The latest, [`1863584`](https://github.com/Shup04/arbitrageAgent/commit/1863584a740636d7830516310f63d7b1b218429d), is titled **“scraping works, ai value detection not quite.”** That is a useful statement of the experiment's stopping point: the collection path had been assembled, while the central question—whether appraisal produces useful leads—remained open.

Several details matter when resuming:

- **Seen does not mean evaluated.** The scraper records every accepted new listing before returning candidates. If it finds more than 20, it returns only the first 10 to limit model calls. The rest are already marked seen and will be excluded next time. For example, 21 new accepted listings produce 10 candidates but 21 seen records. Failed appraisals also have no retry state. This behavior needs resolving before relying on repeated runs.
- **The cost limiter is a prototype.** The active scraper always requests a limit of 150. `get_dynamic_limit()` computes a seven-day average with a 1.5 multiplier and an initial limit of 500, but neither it nor `log_daily_volume()` is called. The ten-item appraisal cap is applied only when there are more than 20 new listings; it is not a general spending budget.
- **Valuation has little supporting evidence.** The model receives the title, asking price, keyword and possibly one image. The script does not retrieve comparable completed sales, full listing details, repair costs or realized resale results. Repair effort appears in the prompt rather than a separate calculation.
- **The shortlist is temporary.** Although listing URLs are retained in memory, the winning list has no durable export. The console's “profitable flips” wording describes model-positive candidates, not confirmed profit.
- **No reproducible environment or evaluation suite is recorded.** Imports identify the Python packages, but package versions are not pinned. Current service compatibility, appraisal quality and cost need to be established in the environment used to resume the experiment.

## Setup and running

Use Python 3 in a separate virtual environment. The repository does not declare a minimum Python version or a tested dependency set. The packages below are derived from the current imports:

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install requests python-dotenv apify-client openai
```

Provide `APIFY_API_TOKEN` and `OPENAI_API_KEY` through your environment or a local `.env` file. Keep real credentials out of Git. Both services must be accessible to the configured accounts, including the actor and model named in the source.

Before running, inspect the search terms, allowed cities, search-area URL and actor limits in `main.py`. Also read the seen-before-evaluated limitation above: the present script changes which listings future runs can see.

From the directory where you want the SQLite database stored:

```sh
python /path/to/arbitrageAgent/main.py
```

This performs live scraping and model requests, can incur service charges, downloads listing images, and creates or updates `arbitrage_memory.db`. There is no dry-run mode. Listing data and available images are sent to the configured external services. Preserve an existing database if its seen-item history matters; deleting it causes those listings to become eligible again. These commands describe the current entry point, not a freshly verified live run.

## Definition of done

A bounded completion target is a repeatable, human-reviewed scouting experiment. It does not require a dashboard, autonomous purchasing or a perpetual background service.

- [ ] **Make processing recoverable.** Distinguish discovery from completed appraisal. Verify that a batch of 21 new listings, a repeated scrape and a failed model call leave every unevaluated listing available for later processing without unnecessarily reappraising completed items.
- [ ] **Set and demonstrate one collection/appraisal budget.** Choose a fixed limit or connect the existing adaptive-volume idea. Record the actual actor/model configuration, dependency versions and observed cost for one representative run.
- [ ] **Evaluate the central idea against a fixed sample.** Include complete items, accessories, damaged goods and missing images. Compare identification and resale estimates with manual review and recorded comparable-sale evidence, using acceptance criteria chosen before examining the results. Record whether the approach is useful, including a negative result if it is not.
- [ ] **Leave a reviewable result.** Preserve candidate URLs, asking prices, appraisal reasoning and review outcomes beyond process exit, clearly distinguishing model-positive candidates from confirmed sales or profit.
- [ ] **Record a final experiment outcome.** Save one representative run and its limitations, and state whether the result is a usable personal scouting tool or a concluded experiment whose valuation approach did not meet the chosen criteria.
