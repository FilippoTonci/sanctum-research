| Setup | Leak recall | Precision | Typed F1 | Typed F2 | Missed (of 487) |
|---|---|---|---|---|---|
| Shipped default: Presidio + en_core_web_sm | 0.79 | 0.74 | 0.61 | 0.63 | 101 |
| Presidio + en_core_web_lg + propagate | 0.89 | 0.76 | 0.72 | 0.77 | 55 |
| Shipped Pro: + gliner_medium @0.4 | 0.87 | 0.89 | 0.85 | 0.85 | 63 |
| gliner_medium @0.2 + propagate | 0.94 | 0.82 | 0.84 | 0.87 | 31 |
| **kg gliner-pii-base @0.2 + propagate** | 0.96 | 0.83 | 0.84 | 0.88 | 18 |
| kg gliner-pii-base ONNX-uint8 @0.2 + propagate | 0.95 | 0.81 | 0.81 | 0.86 | 22 |
| fastino GLiNER2-PII @0.5 + propagate | 0.92 | 0.89 | 0.88 | 0.89 | 41 |
| nvidia gliner-PII @0.4 + propagate | 0.87 | 0.95 | 0.89 | 0.87 | 61 |

| Recall | PERSON | ORGANIZATION | LOCATION | DATE_TIME | EMAIL | PHONE | FINANCIAL | GOV_ID |
|---|---|---|---|---|---|---|---|---|
| Shipped default: Presidio + en_core_web_sm | 0.59 | 0.51 | 0.53 | 0.94 | 0.96 | 0.96 | 0.53 | 0.28 |
| Presidio + en_core_web_lg + propagate | 0.88 | 0.81 | 0.69 | 0.91 | 1.00 | 0.96 | 0.53 | 0.25 |
| Shipped Pro: + gliner_medium @0.4 | 0.86 | 0.86 | 0.98 | 0.86 | 1.00 | 1.00 | 0.63 | 0.35 |
| gliner_medium @0.2 + propagate | 0.98 | 0.92 | 1.00 | 0.88 | 1.00 | 1.00 | 0.68 | 0.38 |
| **kg gliner-pii-base @0.2 + propagate** | 0.97 | 0.88 | 0.86 | 0.89 | 1.00 | 0.87 | 1.00 | 0.72 |
| kg gliner-pii-base ONNX-uint8 @0.2 + propagate | 0.96 | 0.90 | 0.85 | 0.89 | 0.96 | 0.87 | 0.95 | 0.65 |
| fastino GLiNER2-PII @0.5 + propagate | 0.96 | 0.93 | 0.83 | 0.91 | 1.00 | 0.96 | 0.74 | 0.55 |
| nvidia gliner-PII @0.4 + propagate | 0.95 | 0.75 | 0.88 | 0.89 | 1.00 | 0.96 | 0.68 | 0.55 |

| Precision | PERSON | ORGANIZATION | LOCATION | DATE_TIME | EMAIL | PHONE | FINANCIAL | GOV_ID |
|---|---|---|---|---|---|---|---|---|
| Shipped default: Presidio + en_core_web_sm | 0.69 | 0.25 | 0.64 | 0.60 | 1.00 | 0.79 | 0.91 | 1.00 |
| Presidio + en_core_web_lg + propagate | 0.87 | 0.32 | 0.77 | 0.68 | 1.00 | 0.79 | 0.83 | 1.00 |
| Shipped Pro: + gliner_medium @0.4 | 0.90 | 0.66 | 0.95 | 0.96 | 1.00 | 0.74 | 0.75 | 0.74 |
| gliner_medium @0.2 + propagate | 0.80 | 0.57 | 0.91 | 0.95 | 0.90 | 0.70 | 0.54 | 0.75 |
| **kg gliner-pii-base @0.2 + propagate** | 0.97 | 0.52 | 0.84 | 0.93 | 0.96 | 0.87 | 0.47 | 0.74 |
| kg gliner-pii-base ONNX-uint8 @0.2 + propagate | 0.93 | 0.52 | 0.86 | 0.89 | 0.96 | 0.87 | 0.37 | 0.65 |
| fastino GLiNER2-PII @0.5 + propagate | 0.94 | 0.65 | 0.95 | 0.92 | 1.00 | 0.79 | 0.78 | 0.96 |
| nvidia gliner-PII @0.4 + propagate | 0.94 | 0.83 | 0.99 | 0.96 | 1.00 | 0.81 | 0.87 | 0.96 |
