## Leaderboard (hard corpus, 487 entities; thresholds 2-fold cross-validated on F2)

| Config | Leak recall | Char recall | Precision | Typed F2 | Typed F1 | Missed | Thr (fold A/B) | Sanctum-fixtures F1 | ms / 1k chars | Weights MB | Peak RAM MB | Licence |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| presidio+nvidia_gliner_pii +propagate | 0.91 | 0.89 | 0.90 | **0.88** | 0.88 | 46 | 0.15/0.25 | 0.89 | 357 | 1793 | 4112 | NVIDIA Open Model License |
| presidio+nvidia_gliner_pii | 0.90 | 0.89 | 0.92 | **0.88** | 0.88 | 51 | 0.15/0.15 | 0.89 | 357 | 1793 | 4112 | NVIDIA Open Model License |
| presidio+kg_pii_base +propagate | 0.94 | 0.94 | 0.85 | **0.88** | 0.85 | 27 | 0.2/0.25 | 0.94 | 111 | 675 | 2324 | Apache-2.0 |
| presidio+fastino_gliner2_pii +propagate | 0.91 | 0.84 | 0.86 | **0.88** | 0.86 | 44 | 0.6/0.3 | 0.91 | 179 | 1244 | 3166 | Apache-2.0 |
| presidio+gliner_medium (shipped Pro) +propagate | 0.93 | 0.91 | 0.82 | **0.87** | 0.84 | 36 | 0.2/0.3 | 0.88 | 142 | 781 | 2056 | Apache-2.0 |
| presidio+gliner_multi_pii +propagate | 0.94 | 0.95 | 0.86 | **0.87** | 0.85 | 30 | 0.15/0.2 | 0.84 | 169 | 1156 | 2929 | Apache-2.0 |
| nvidia_gliner_pii | 0.89 | 0.87 | 0.92 | **0.87** | 0.88 | 56 | 0.15/0.15 | 0.89 | 331 | 1793 | 4110 | NVIDIA Open Model License |
| presidio+gliner_medium (shipped Pro) | 0.93 | 0.92 | 0.81 | **0.87** | 0.83 | 35 | 0.15/0.15 | 0.89 | 142 | 781 | 2056 | Apache-2.0 |
| presidio+fastino_gliner2_pii | 0.89 | 0.84 | 0.87 | **0.86** | 0.86 | 54 | 0.5/0.3 | 0.92 | 179 | 1244 | 3166 | Apache-2.0 |
| fastino_gliner2_pii | 0.88 | 0.85 | 0.86 | **0.86** | 0.85 | 59 | 0.2/0.3 | 0.91 | 157 | 1244 | 3166 | Apache-2.0 |
| presidio+kg_pii_base | 0.92 | 0.93 | 0.86 | **0.86** | 0.84 | 41 | 0.2/0.25 | 0.94 | 111 | 675 | 2324 | Apache-2.0 |
| presidio+kg_pii_base (sanctum prompts) +propagate | 0.93 | 0.88 | 0.84 | **0.85** | 0.83 | 32 | 0.2/0.25 | 0.89 | 109 | 675 | 2241 | Apache-2.0 |
| kg_pii_base | 0.95 | 0.95 | 0.80 | **0.85** | 0.81 | 23 | 0.15/0.15 | 0.94 | 86 | 675 | 2129 | Apache-2.0 |
| presidio+kg_pii_base_onnx_q8 +propagate | 0.94 | 0.92 | 0.85 | **0.85** | 0.82 | 31 | 0.25/0.2 | 0.91 | 100 | 205 | 1309 | Apache-2.0 |
| presidio+kg_pii_base (sanctum prompts) | 0.93 | 0.89 | 0.84 | **0.85** | 0.82 | 34 | 0.15/0.25 | 0.89 | 109 | 675 | 2241 | Apache-2.0 |
| presidio+kg_pii_base_onnx_q8 | 0.93 | 0.91 | 0.84 | **0.85** | 0.82 | 36 | 0.2/0.2 | 0.91 | 100 | 205 | 1309 | Apache-2.0 |
| presidio+gliner_multi_pii | 0.91 | 0.94 | 0.87 | **0.85** | 0.84 | 46 | 0.15/0.15 | 0.85 | 169 | 1156 | 2929 | Apache-2.0 |
| presidio+kg_pii_small +propagate | 0.96 | 0.92 | 0.78 | **0.84** | 0.79 | 20 | 0.3/0.3 | 0.81 | 110 | 330 | 1897 | Apache-2.0 |
| gliner_medium_v21 | 0.93 | 0.91 | 0.80 | **0.83** | 0.79 | 34 | 0.15/0.15 | 0.85 | 122 | 781 | 2056 | Apache-2.0 |
| presidio+trf +propagate | 0.90 | 0.79 | 0.84 | **0.83** | 0.81 | 50 | n/a | 0.81 | 75 | 501 | 3067 | MIT |
| presidio+trf | 0.89 | 0.78 | 0.85 | **0.83** | 0.82 | 55 | n/a | 0.81 | 75 | 501 | 3067 | MIT |
| e3jsi_multi_pii_domains | 0.84 | 0.89 | 0.89 | **0.82** | 0.83 | 76 | 0.15/0.15 | 0.85 | 147 | 1177 | 3024 | Apache-2.0 |
| presidio+kg_pii_small | 0.94 | 0.92 | 0.73 | **0.81** | 0.75 | 29 | 0.2/0.3 | 0.82 | 110 | 330 | 1897 | Apache-2.0 |
| presidio+kg_pii_small_onnx_q8 +propagate | 0.93 | 0.88 | 0.84 | **0.81** | 0.79 | 35 | 0.3/0.3 | 0.83 | 115 | 86 | 981 | Apache-2.0 |
| bardsai_eu_pii | 0.83 | 0.84 | 0.95 | **0.80** | 0.82 | 81 | 0.2/0.15 | 0.71 | 51 | 1127 | 1222 | Apache-2.0 |
| gliner_multi_pii_v1 | 0.85 | 0.90 | 0.87 | **0.80** | 0.81 | 73 | 0.15/0.15 | 0.85 | 145 | 1156 | 2929 | Apache-2.0 |
| presidio+kg_pii_edge +propagate | 0.90 | 0.87 | 0.79 | **0.80** | 0.76 | 47 | 0.4/0.4 | 0.64 | 86 | 184 | 1633 | Apache-2.0 |
| kg_pii_small | 0.93 | 0.90 | 0.73 | **0.79** | 0.73 | 32 | 0.2/0.3 | 0.80 | 86 | 330 | 1793 | Apache-2.0 |
| presidio+kg_pii_small_onnx_q8 | 0.89 | 0.87 | 0.81 | **0.78** | 0.76 | 52 | 0.3/0.25 | 0.84 | 115 | 86 | 981 | Apache-2.0 |
| presidio+kg_pii_edge | 0.86 | 0.84 | 0.80 | **0.77** | 0.75 | 68 | 0.4/0.4 | 0.64 | 86 | 184 | 1633 | Apache-2.0 |
| presidio+lg +propagate | 0.89 | 0.81 | 0.76 | **0.77** | 0.72 | 55 | n/a | 0.68 | 26 | 445 | 1142 | MIT |
| presidio+lg | 0.86 | 0.79 | 0.78 | **0.75** | 0.72 | 70 | n/a | 0.69 | 26 | 445 | 1142 | MIT |
| kg_pii_edge | 0.84 | 0.81 | 0.80 | **0.74** | 0.73 | 78 | 0.4/0.4 | 0.61 | 61 | 184 | 1524 | Apache-2.0 |
| gliner_small_v21 | 0.87 | 0.84 | 0.80 | **0.74** | 0.71 | 61 | 0.2/0.15 | 0.74 | 72 | 611 | 2061 | Apache-2.0 |
| presidio+kg_pii_edge_onnx_q8 +propagate | 0.92 | 0.87 | 0.72 | **0.73** | 0.67 | 39 | 0.25/0.25 | 0.70 | 89 | 49 | 900 | Apache-2.0 |
| presidio+kg_pii_edge_onnx_q8 | 0.91 | 0.87 | 0.74 | **0.73** | 0.68 | 43 | 0.25/0.25 | 0.70 | 89 | 49 | 900 | Apache-2.0 |
| soelmgd_bert_pii | 0.84 | 0.70 | 0.90 | **0.70** | 0.70 | 77 | 0.2/0.2 | 0.83 | 27 | 267 | 601 | MIT |
| presidio+sm +propagate | 0.84 | 0.77 | 0.73 | **0.66** | 0.62 | 79 | n/a | 0.66 | 24 | 15 | 500 | MIT |
| kg_pii_edge_onnx_q8 | 0.90 | 0.86 | 0.69 | **0.65** | 0.59 | 49 | 0.25/0.2 | 0.59 | 61 | 49 | 760 | Apache-2.0 |
| presidio+openai_privacy_filter | 0.73 | 0.71 | 0.97 | **0.64** | 0.69 | 133 | 0.15/0.15 | 0.72 | 326 | 2827 | 9370 | Apache-2.0 |
| presidio+openai_privacy_filter +propagate | 0.73 | 0.71 | 0.96 | **0.64** | 0.69 | 131 | 0.7/0.15 | 0.71 | 326 | 2827 | 9370 | Apache-2.0 |
| presidio+gretel_bi_small +propagate | 0.63 | 0.63 | 0.99 | **0.64** | 0.72 | 179 | 0.15/0.15 | 0.81 | 102 | 768 | 2258 | Apache-2.0 |
| presidio+sm | 0.79 | 0.75 | 0.74 | **0.63** | 0.61 | 101 | n/a | 0.66 | 24 | 15 | 500 | MIT |
| openai_privacy_filter | 0.69 | 0.66 | 0.97 | **0.60** | 0.66 | 151 | 0.15/0.15 | 0.73 | 291 | 2827 | 9199 | Apache-2.0 |
| presidio+gretel_bi_small | 0.57 | 0.60 | 0.99 | **0.59** | 0.68 | 209 | 0.15/0.15 | 0.74 | 102 | 768 | 2258 | Apache-2.0 |
| dslim_bert_ner | 0.53 | 0.47 | 0.82 | **0.51** | 0.57 | 231 | 0.6/0.4 | 0.61 | 56 | 434 | 791 | MIT |
| gretel_bi_small | 0.48 | 0.50 | 0.98 | **0.50** | 0.60 | 253 | 0.15/0.15 | 0.68 | 79 | 768 | 2149 | Apache-2.0 |

## At each model's default threshold (no tuning)

| Config | Thr | Leak recall | Precision | Typed F1 | Missed |
|---|---|---|---|---|---|
| presidio+nvidia_gliner_pii +propagate | 0.4 | 0.87 | 0.95 | 0.89 | 61 |
| presidio+nvidia_gliner_pii | 0.4 | 0.85 | 0.96 | 0.89 | 75 |
| presidio+fastino_gliner2_pii +propagate | 0.5 | 0.92 | 0.89 | 0.88 | 41 |
| presidio+fastino_gliner2_pii | 0.5 | 0.88 | 0.91 | 0.87 | 57 |
| nvidia_gliner_pii | 0.4 | 0.81 | 0.96 | 0.87 | 91 |
| fastino_gliner2_pii | 0.5 | 0.86 | 0.91 | 0.87 | 69 |
| presidio+gliner_medium (shipped Pro) +propagate | 0.4 | 0.91 | 0.87 | 0.86 | 46 |
| presidio+kg_pii_base +propagate | 0.3 | 0.92 | 0.88 | 0.86 | 37 |
| presidio+kg_pii_base_onnx_q8 +propagate | 0.3 | 0.89 | 0.91 | 0.85 | 54 |
| presidio+gliner_multi_pii +propagate | 0.4 | 0.89 | 0.90 | 0.85 | 56 |
| presidio+gliner_medium (shipped Pro) | 0.4 | 0.87 | 0.89 | 0.85 | 63 |
| presidio+kg_pii_base | 0.3 | 0.89 | 0.90 | 0.85 | 52 |
| presidio+kg_pii_base (sanctum prompts) +propagate | 0.3 | 0.89 | 0.90 | 0.84 | 53 |
| presidio+kg_pii_base (sanctum prompts) | 0.3 | 0.86 | 0.92 | 0.84 | 66 |
| kg_pii_base | 0.3 | 0.87 | 0.90 | 0.84 | 62 |
| presidio+gliner_multi_pii | 0.4 | 0.84 | 0.92 | 0.84 | 80 |
| presidio+kg_pii_base_onnx_q8 | 0.3 | 0.84 | 0.92 | 0.83 | 78 |
| e3jsi_multi_pii_domains | 0.4 | 0.75 | 0.94 | 0.82 | 121 |
| presidio+trf | none | 0.89 | 0.85 | 0.82 | 55 |
| bardsai_eu_pii | 0.4 | 0.83 | 0.95 | 0.82 | 83 |
| presidio+trf +propagate | none | 0.90 | 0.84 | 0.81 | 50 |
| gliner_medium_v21 | 0.4 | 0.81 | 0.88 | 0.80 | 91 |
| gliner_multi_pii_v1 | 0.4 | 0.76 | 0.91 | 0.79 | 116 |
| presidio+kg_pii_small +propagate | 0.3 | 0.96 | 0.78 | 0.79 | 20 |
| presidio+kg_pii_small_onnx_q8 +propagate | 0.3 | 0.93 | 0.84 | 0.79 | 35 |
| presidio+kg_pii_small | 0.3 | 0.92 | 0.80 | 0.78 | 38 |
| presidio+kg_pii_small_onnx_q8 | 0.3 | 0.89 | 0.85 | 0.77 | 54 |
| kg_pii_small | 0.3 | 0.91 | 0.80 | 0.76 | 43 |
| presidio+lg +propagate | none | 0.89 | 0.76 | 0.72 | 55 |
| gliner_small_v21 | 0.4 | 0.71 | 0.88 | 0.72 | 139 |
| presidio+lg | none | 0.86 | 0.78 | 0.72 | 70 |
| soelmgd_bert_pii | 0.4 | 0.76 | 0.93 | 0.70 | 118 |
| presidio+openai_privacy_filter | 0.4 | 0.73 | 0.97 | 0.69 | 133 |
| presidio+openai_privacy_filter +propagate | 0.4 | 0.74 | 0.96 | 0.69 | 125 |
| presidio+kg_pii_edge_onnx_q8 +propagate | 0.3 | 0.85 | 0.80 | 0.69 | 74 |
| presidio+kg_pii_edge_onnx_q8 | 0.3 | 0.81 | 0.82 | 0.69 | 92 |
| openai_privacy_filter | 0.4 | 0.69 | 0.97 | 0.66 | 151 |
| presidio+kg_pii_edge +propagate | 0.3 | 0.98 | 0.61 | 0.64 | 11 |
| presidio+gretel_bi_small +propagate | 0.4 | 0.53 | 0.99 | 0.64 | 230 |
| presidio+kg_pii_edge | 0.3 | 0.96 | 0.62 | 0.64 | 21 |
| kg_pii_edge | 0.3 | 0.95 | 0.62 | 0.63 | 22 |
| presidio+sm +propagate | none | 0.84 | 0.73 | 0.62 | 79 |
| kg_pii_edge_onnx_q8 | 0.3 | 0.75 | 0.81 | 0.61 | 120 |
| presidio+sm | none | 0.79 | 0.74 | 0.61 | 101 |
| presidio+gretel_bi_small | 0.4 | 0.48 | 0.99 | 0.61 | 253 |
| dslim_bert_ner | 0.4 | 0.53 | 0.82 | 0.57 | 227 |
| gretel_bi_small | 0.4 | 0.37 | 0.99 | 0.51 | 307 |

## Recall by entity group (cross-validated threshold)

| Config | PERSON | ORGANIZATION | LOCATION | DATE_TIME | GOV_ID | FINANCIAL |
|---|---|---|---|---|---|---|
| presidio+nvidia_gliner_pii +propagate | 0.96 | 0.81 | 0.93 | 0.94 | 0.55 | 0.68 |
| presidio+nvidia_gliner_pii | 0.91 | 0.78 | 0.97 | 0.95 | 0.55 | 0.79 |
| presidio+kg_pii_base +propagate | 0.97 | 0.88 | 0.86 | 0.88 | 0.65 | 1.00 |
| presidio+fastino_gliner2_pii +propagate | 0.95 | 0.93 | 0.80 | 0.92 | 0.55 | 0.74 |
| presidio+gliner_medium (shipped Pro) +propagate | 0.97 | 0.92 | 1.00 | 0.88 | 0.35 | 0.68 |
| presidio+gliner_multi_pii +propagate | 0.97 | 0.90 | 0.93 | 0.82 | 0.42 | 1.00 |
| nvidia_gliner_pii | 0.91 | 0.78 | 0.97 | 0.95 | 0.47 | 0.79 |
| presidio+gliner_medium (shipped Pro) | 0.94 | 0.92 | 1.00 | 0.88 | 0.40 | 0.74 |
| presidio+fastino_gliner2_pii | 0.88 | 0.92 | 0.81 | 0.93 | 0.55 | 0.74 |
| fastino_gliner2_pii | 0.89 | 0.99 | 0.83 | 0.95 | 0.42 | 0.74 |
| presidio+kg_pii_base | 0.88 | 0.88 | 0.86 | 0.88 | 0.65 | 1.00 |
| presidio+kg_pii_base (sanctum prompts) +propagate | 0.97 | 0.92 | 0.71 | 0.85 | 0.55 | 1.00 |
| kg_pii_base | 0.90 | 0.90 | 0.86 | 0.92 | 0.72 | 0.95 |
| presidio+kg_pii_base_onnx_q8 +propagate | 0.93 | 0.89 | 0.83 | 0.87 | 0.62 | 0.95 |
| presidio+kg_pii_base (sanctum prompts) | 0.93 | 0.92 | 0.75 | 0.85 | 0.57 | 1.00 |
| presidio+kg_pii_base_onnx_q8 | 0.88 | 0.90 | 0.85 | 0.89 | 0.65 | 0.95 |
| presidio+gliner_multi_pii | 0.87 | 0.88 | 0.93 | 0.84 | 0.42 | 1.00 |
| presidio+kg_pii_small +propagate | 0.92 | 0.75 | 0.92 | 0.92 | 0.72 | 0.68 |
| gliner_medium_v21 | 0.95 | 0.92 | 1.00 | 0.92 | 0.35 | 0.68 |
| presidio+trf +propagate | 0.90 | 0.82 | 0.80 | 0.98 | 0.28 | 0.58 |
| presidio+trf | 0.88 | 0.82 | 0.80 | 0.98 | 0.28 | 0.58 |
| e3jsi_multi_pii_domains | 0.86 | 0.88 | 0.86 | 0.74 | 0.57 | 0.63 |
| presidio+kg_pii_small | 0.86 | 0.75 | 0.93 | 0.93 | 0.72 | 0.68 |
| presidio+kg_pii_small_onnx_q8 +propagate | 0.90 | 0.75 | 0.81 | 0.80 | 0.70 | 0.74 |
| bardsai_eu_pii | 0.98 | 0.89 | 0.85 | 0.19 | 0.88 | 0.68 |
| gliner_multi_pii_v1 | 0.87 | 0.88 | 0.93 | 0.60 | 0.35 | 0.95 |
| presidio+kg_pii_edge +propagate | 0.89 | 0.75 | 0.80 | 0.80 | 0.62 | 0.84 |
| kg_pii_small | 0.86 | 0.75 | 0.93 | 0.92 | 0.70 | 0.68 |
| presidio+kg_pii_small_onnx_q8 | 0.78 | 0.76 | 0.81 | 0.81 | 0.70 | 0.74 |
| presidio+kg_pii_edge | 0.77 | 0.74 | 0.80 | 0.80 | 0.62 | 0.84 |
| presidio+lg +propagate | 0.88 | 0.81 | 0.69 | 0.91 | 0.25 | 0.53 |
| presidio+lg | 0.81 | 0.79 | 0.69 | 0.91 | 0.25 | 0.53 |
| kg_pii_edge | 0.77 | 0.74 | 0.80 | 0.78 | 0.57 | 0.63 |
| gliner_small_v21 | 0.94 | 0.93 | 0.95 | 0.82 | 0.25 | 0.26 |
| presidio+kg_pii_edge_onnx_q8 +propagate | 0.79 | 0.79 | 0.90 | 0.65 | 0.60 | 0.74 |
| presidio+kg_pii_edge_onnx_q8 | 0.76 | 0.79 | 0.90 | 0.65 | 0.60 | 0.74 |
| soelmgd_bert_pii | 0.86 | 0.14 | 0.81 | 0.94 | 0.72 | 0.00 |
| presidio+sm +propagate | 0.69 | 0.56 | 0.53 | 0.94 | 0.28 | 0.53 |
| kg_pii_edge_onnx_q8 | 0.77 | 0.79 | 0.92 | 0.58 | 0.50 | 0.68 |
| presidio+openai_privacy_filter | 0.92 | 0.00 | 0.37 | 0.73 | 0.03 | 0.89 |
| presidio+openai_privacy_filter +propagate | 0.94 | 0.00 | 0.36 | 0.69 | 0.03 | 0.89 |
| presidio+gretel_bi_small +propagate | 0.77 | 0.17 | 0.49 | 0.47 | 0.42 | 0.68 |
| presidio+sm | 0.59 | 0.51 | 0.53 | 0.94 | 0.28 | 0.53 |
| openai_privacy_filter | 0.92 | 0.00 | 0.37 | 0.67 | 0.00 | 0.68 |
| presidio+gretel_bi_small | 0.62 | 0.12 | 0.49 | 0.47 | 0.42 | 0.68 |
| dslim_bert_ner | 0.81 | 0.81 | 0.83 | 0.00 | 0.00 | 0.00 |
| gretel_bi_small | 0.62 | 0.12 | 0.49 | 0.25 | 0.35 | 0.53 |

## Precision by entity group (cross-validated threshold)

| Config | PERSON | ORGANIZATION | LOCATION | DATE_TIME | GOV_ID | FINANCIAL |
|---|---|---|---|---|---|---|
| presidio+nvidia_gliner_pii +propagate | 0.90 | 0.72 | 0.96 | 0.94 | 0.85 | 0.87 |
| presidio+nvidia_gliner_pii | 0.91 | 0.75 | 0.95 | 0.94 | 0.85 | 0.88 |
| presidio+kg_pii_base +propagate | 0.97 | 0.57 | 0.84 | 0.93 | 0.74 | 0.49 |
| presidio+fastino_gliner2_pii +propagate | 0.94 | 0.55 | 0.93 | 0.92 | 0.92 | 0.82 |
| presidio+gliner_medium (shipped Pro) +propagate | 0.80 | 0.57 | 0.93 | 0.95 | 0.74 | 0.62 |
| presidio+gliner_multi_pii +propagate | 0.88 | 0.65 | 0.91 | 0.91 | 0.85 | 0.59 |
| nvidia_gliner_pii | 0.90 | 0.75 | 0.95 | 0.94 | 0.79 | 0.83 |
| presidio+gliner_medium (shipped Pro) | 0.79 | 0.58 | 0.90 | 0.94 | 0.80 | 0.52 |
| presidio+fastino_gliner2_pii | 0.96 | 0.59 | 0.93 | 0.91 | 0.92 | 0.78 |
| fastino_gliner2_pii | 0.96 | 0.56 | 0.92 | 0.92 | 0.87 | 0.61 |
| presidio+kg_pii_base | 0.98 | 0.62 | 0.84 | 0.93 | 0.74 | 0.49 |
| presidio+kg_pii_base (sanctum prompts) +propagate | 0.92 | 0.55 | 0.84 | 0.96 | 0.85 | 0.47 |
| kg_pii_base | 0.95 | 0.50 | 0.82 | 0.92 | 0.69 | 0.38 |
| presidio+kg_pii_base_onnx_q8 +propagate | 0.97 | 0.56 | 0.86 | 0.90 | 0.66 | 0.38 |
| presidio+kg_pii_base (sanctum prompts) | 0.89 | 0.58 | 0.81 | 0.96 | 0.85 | 0.45 |
| presidio+kg_pii_base_onnx_q8 | 0.96 | 0.59 | 0.86 | 0.89 | 0.65 | 0.37 |
| presidio+gliner_multi_pii | 0.89 | 0.68 | 0.89 | 0.91 | 0.85 | 0.59 |
| presidio+kg_pii_small +propagate | 0.85 | 0.52 | 0.89 | 0.82 | 0.54 | 0.31 |
| gliner_medium_v21 | 0.72 | 0.52 | 0.88 | 0.92 | 0.58 | 0.59 |
| presidio+trf +propagate | 0.96 | 0.51 | 0.90 | 0.67 | 1.00 | 0.85 |
| presidio+trf | 0.98 | 0.57 | 0.90 | 0.67 | 1.00 | 0.85 |
| e3jsi_multi_pii_domains | 0.92 | 0.66 | 0.92 | 0.91 | 0.77 | 0.71 |
| presidio+kg_pii_small | 0.82 | 0.50 | 0.88 | 0.75 | 0.52 | 0.21 |
| presidio+kg_pii_small_onnx_q8 +propagate | 0.92 | 0.58 | 0.87 | 0.93 | 0.52 | 0.32 |
| bardsai_eu_pii | 0.99 | 0.59 | 0.93 | 1.00 | 0.76 | 0.89 |
| gliner_multi_pii_v1 | 0.89 | 0.66 | 0.89 | 0.88 | 0.82 | 0.58 |
| presidio+kg_pii_edge +propagate | 0.96 | 0.52 | 0.89 | 0.93 | 0.61 | 0.18 |
| kg_pii_small | 0.78 | 0.50 | 0.88 | 0.75 | 0.51 | 0.21 |
| presidio+kg_pii_small_onnx_q8 | 0.90 | 0.56 | 0.85 | 0.92 | 0.47 | 0.31 |
| presidio+kg_pii_edge | 0.97 | 0.61 | 0.89 | 0.93 | 0.61 | 0.18 |
| presidio+lg +propagate | 0.87 | 0.32 | 0.77 | 0.68 | 1.00 | 0.83 |
| presidio+lg | 0.88 | 0.35 | 0.77 | 0.68 | 1.00 | 0.83 |
| kg_pii_edge | 0.97 | 0.60 | 0.89 | 0.93 | 0.56 | 0.14 |
| gliner_small_v21 | 0.72 | 0.44 | 0.79 | 0.96 | 0.40 | 0.50 |
| presidio+kg_pii_edge_onnx_q8 +propagate | 0.92 | 0.35 | 0.70 | 0.88 | 0.48 | 0.29 |
| presidio+kg_pii_edge_onnx_q8 | 0.91 | 0.39 | 0.70 | 0.88 | 0.48 | 0.29 |
| soelmgd_bert_pii | 0.94 | 0.58 | 0.72 | 0.75 | 0.43 | 0.00 |
| presidio+sm +propagate | 0.71 | 0.23 | 0.64 | 0.60 | 1.00 | 0.91 |
| kg_pii_edge_onnx_q8 | 0.86 | 0.34 | 0.68 | 0.84 | 0.35 | 0.28 |
| presidio+openai_privacy_filter | 0.94 | 0.00 | 1.00 | 0.97 | 1.00 | 0.26 |
| presidio+openai_privacy_filter +propagate | 0.89 | 0.00 | 1.00 | 0.97 | 1.00 | 0.26 |
| presidio+gretel_bi_small +propagate | 0.95 | 1.00 | 0.84 | 0.98 | 0.89 | 0.93 |
| presidio+sm | 0.69 | 0.25 | 0.64 | 0.60 | 1.00 | 0.91 |
| openai_privacy_filter | 0.93 | 0.00 | 1.00 | 0.97 | 0.00 | 0.20 |
| presidio+gretel_bi_small | 0.98 | 1.00 | 0.84 | 0.98 | 0.89 | 0.93 |
| dslim_bert_ner | 0.95 | 0.39 | 0.81 | 0.00 | 0.00 | 0.00 |
| gretel_bi_small | 0.98 | 1.00 | 0.84 | 0.95 | 0.88 | 0.91 |
