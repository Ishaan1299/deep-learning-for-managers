# Food_Delivery — Raw Zomato Source Data

Zomato restaurant dataset files consumed by `food_delivery_sentiment/data_prep.py`.

## Files

| File | What | When to read |
|---|---|---|
| `zomato.csv` | 9,551 restaurants worldwide (8,652 India, Country Code=1); columns: Restaurant ID/Name, City, Cuisines, Average Cost for two, Price range, Has Table booking, Has Online delivery, Aggregate rating, Rating text, Votes | Inspecting raw CSV structure, checking column names before modifying data_prep.py |
| `file1.json` | Zomato API response — 479 pages of restaurant listings, primarily New Delhi NCR; each record has name, cuisines, price_range, avg_cost_for_two, has_online_delivery, has_table_booking, location, user_rating (rating_text + aggregate_rating + votes) | Inspecting JSON structure, debugging JSON parsing in data_prep.py |
| `file2.json` | Zomato API response — 550 pages, New Delhi NCR | Inspecting JSON structure |
| `file3.json` | Zomato API response — 550 pages, New Delhi NCR | Inspecting JSON structure |
| `file4.json` | Zomato API response — 550 pages, New Delhi NCR | Inspecting JSON structure |
| `file5.json` | Zomato API response — 550 pages, New Delhi NCR | Inspecting JSON structure |
| `Country-Code.xlsx` | Mapping of Zomato Country Code integers to country names (Code 1 = India) | Verifying country filter applied in data_prep.py |

## Usage Note

All files are **read-only source data** consumed by `../food_delivery_sentiment/data_prep.py`. JSON files cover Delhi NCR; CSV covers all India. Combined, they yield 32,912 rated India restaurants after filtering "Not rated" entries.
