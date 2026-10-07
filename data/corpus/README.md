# Hand-annotated hard corpus

Synthetic but realistic legal / consulting documents written for this
benchmark. All people, companies, numbers and addresses are invented;
any resemblance to real entities is coincidental. IBANs, card numbers and
SSNs are structurally plausible but fake.

## Markup

* `[[TYPE|text]]` — a ground-truth entity. Must be found.
* `[[?TYPE|text]]` — an *optional* entity (borderline: courts, public
  agencies, relative dates, public figures). Predictions overlapping an
  optional span are neither rewarded nor penalised.

`bench/corpus.py` strips the markup and produces character offsets.

## Types

Sanctum's taxonomy plus one extra bucket:

| Type | What counts |
|---|---|
| PERSON | Names of people (titles like "Dr." excluded). First-name-only mentions count. |
| ORGANIZATION | Companies, law firms, banks, hospitals, universities, NGOs. |
| LOCATION | Street addresses (one span for the whole address), cities, regions, countries. |
| DATE_TIME | Calendar dates, dates of birth, specific times. Durations ("30 days") are not PII. |
| EMAIL_ADDRESS, PHONE_NUMBER, URL, IP_ADDRESS | as named |
| IBAN_CODE, CREDIT_CARD, US_SSN, US_BANK_NUMBER, US_DRIVER_LICENSE | as named |
| ID_NUMBER | Any other personal identifier: passport, national ID, tax code, NHS / medical record number, employee ID, case-specific account refs. Not in Sanctum's taxonomy today — tracked to show coverage gaps. |

Hard negatives (should **not** be flagged) are sprinkled throughout:
product names, statute and clause references, monetary amounts, version
numbers, role nouns ("the Client"), and capitalised common words.
