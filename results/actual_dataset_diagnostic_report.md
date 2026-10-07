| Attack_Class      |   Test_Samples |   Precision |   Recall |   F1_Score | Status                         |
|:------------------|---------------:|------------:|---------:|-----------:|:-------------------------------|
| benign            |            900 |      100    |    99.22 |      99.61 | [EXCELLENT] Flawless Detection |
| dos-http          |            900 |       78.44 |    86.89 |      82.45 | [MODERATE] Minor Confusion     |
| dos-icmp          |            900 |      100    |    99.89 |      99.94 | [EXCELLENT] Flawless Detection |
| dos-tcp-syn       |            900 |      100    |   100    |     100    | [EXCELLENT] Flawless Detection |
| dos-udp           |            900 |       99.89 |   100    |      99.94 | [EXCELLENT] Flawless Detection |
| infog-scan-os     |            301 |       99.6  |    83.06 |      90.58 | [GOOD] High Accuracy           |
| infog-scan-port   |            900 |       90.49 |    96.22 |      93.27 | [GOOD] High Accuracy           |
| infog-scan-vuln   |            900 |       99.08 |    95.89 |      97.46 | [EXCELLENT] Flawless Detection |
| inject-http       |            900 |       95.36 |    84.56 |      89.63 | [GOOD] High Accuracy           |
| inject-sql        |            900 |       90.12 |    88.22 |      89.16 | [GOOD] High Accuracy           |
| inject-xss        |            900 |       82.6  |    91.78 |      86.95 | [GOOD] High Accuracy           |
| malwr-backd-tcp   |            900 |      100    |    96.11 |      98.02 | [EXCELLENT] Flawless Detection |
| malwr-brutef-http |            900 |       92.29 |    83.78 |      87.83 | [GOOD] High Accuracy           |
| malwr-ransomware  |            900 |      100    |    87.78 |      93.49 | [GOOD] High Accuracy           |
| mitm-arpspoof     |            364 |       66.79 |   100    |      80.09 | [MODERATE] Minor Confusion     |