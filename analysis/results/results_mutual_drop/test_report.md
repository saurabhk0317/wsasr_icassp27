# <star> placement vs validator disagreement (test)

Excluded char classes: punct. Star measures: see analyze.py docstring.

## bh (104293 chars, 1874 disputed = 1.80%)

| model | unit | measure | mean(disputed) | mean(undisputed) | ratio | AUC |
|---|---|---|---|---|---|---|
| ctc | char | p_tok_w0 | 0.315 | 0.019 | 16.7 | 0.857 |
| ctc | char | neg_lam | -1.652 | -5.471 | - | 0.871 |
| ctc | word | p_word_w0 | 0.032 | 0.005 | 6.9 | 0.535 |
| ctc | word | neg_lam | -32.561 | -34.929 | - | 0.545 |
| ctc | bound | ins_w0 | 0.288 | 0.014 | 20.9 | 0.798 |
| otc_recip | char | p_tok_w0 | 0.410 | 0.020 | 20.2 | 0.854 |
| otc_recip | char | neg_lam | 0.059 | -7.554 | - | 0.869 |
| otc_recip | char | own_p | 0.397 | 0.018 | 22.5 | 0.852 |
| otc_recip | word | p_word_w0 | 0.049 | 0.005 | 9.4 | 0.542 |
| otc_recip | word | neg_lam | -37.999 | -41.297 | - | 0.554 |
| otc_recip | bound | ins_w0 | 0.379 | 0.025 | 15.3 | 0.803 |
| otc_token | char | p_tok_w0 | 0.400 | 0.019 | 21.2 | 0.875 |
| otc_token | char | neg_lam | -0.218 | -7.147 | - | 0.885 |
| otc_token | char | own_p | 0.391 | 0.017 | 22.4 | 0.875 |
| otc_token | word | p_word_w0 | 0.044 | 0.005 | 9.3 | 0.536 |
| otc_token | word | neg_lam | -37.947 | -40.472 | - | 0.544 |
| otc_token | bound | ins_w0 | 0.331 | 0.022 | 15.3 | 0.796 |
| otc_token_word | char | p_tok_w0 | 0.402 | 0.020 | 20.1 | 0.876 |
| otc_token_word | char | neg_lam | -0.041 | -6.977 | - | 0.888 |
| otc_token_word | char | own_p | 0.362 | 0.016 | 22.7 | 0.847 |
| otc_token_word | word | p_word_w0 | 0.052 | 0.007 | 7.2 | 0.542 |
| otc_token_word | word | neg_lam | -35.490 | -39.208 | - | 0.555 |
| otc_token_word | word | own_p | 0.025 | 0.005 | 5.2 | 0.531 |
| otc_token_word | bound | ins_w0 | 0.351 | 0.022 | 16.2 | 0.789 |
| otc_token_word_recip | char | p_tok_w0 | 0.393 | 0.020 | 20.0 | 0.852 |
| otc_token_word_recip | char | neg_lam | -0.302 | -7.404 | - | 0.866 |
| otc_token_word_recip | char | own_p | 0.351 | 0.016 | 22.3 | 0.826 |
| otc_token_word_recip | word | p_word_w0 | 0.050 | 0.006 | 8.9 | 0.542 |
| otc_token_word_recip | word | neg_lam | -37.465 | -40.520 | - | 0.548 |
| otc_token_word_recip | word | own_p | 0.019 | 0.002 | 8.3 | 0.519 |
| otc_token_word_recip | bound | ins_w0 | 0.376 | 0.024 | 15.7 | 0.789 |
| otc_word | char | p_tok_w0 | 0.321 | 0.018 | 17.6 | 0.824 |
| otc_word | char | neg_lam | -1.551 | -6.582 | - | 0.850 |
| otc_word | word | p_word_w0 | 0.033 | 0.004 | 8.9 | 0.536 |
| otc_word | word | neg_lam | -33.915 | -35.962 | - | 0.544 |
| otc_word | word | own_p | 0.043 | 0.005 | 8.1 | 0.547 |
| otc_word | bound | ins_w0 | 0.376 | 0.021 | 17.8 | 0.828 |
| otc_word_tuned | char | p_tok_w0 | 0.326 | 0.020 | 16.3 | 0.853 |
| otc_word_tuned | char | neg_lam | -1.463 | -5.544 | - | 0.869 |
| otc_word_tuned | word | p_word_w0 | 0.041 | 0.007 | 5.8 | 0.540 |
| otc_word_tuned | word | neg_lam | -33.267 | -36.997 | - | 0.557 |
| otc_word_tuned | word | own_p | 0.053 | 0.008 | 6.4 | 0.562 |
| otc_word_tuned | bound | ins_w0 | 0.297 | 0.015 | 20.3 | 0.785 |

**Share of units with star > 0.5** (disputed / undisputed; and how many of all starred units are disputed)

| model | char: disputed | char: undisputed | char: precision | word (p_word_w0): disputed | word: undisputed | boundary ins (w=0): disputed | boundary: undisputed |
|---|---|---|---|---|---|---|---|
| ctc | 0.300 (562) | 0.0092 (941) | 0.374 | 0.031 | 0.0043 | 0.240 | 0.0022 |
| otc_recip | 0.405 (759) | 0.0154 (1579) | 0.325 | 0.049 | 0.0050 | 0.337 | 0.0055 |
| otc_token | 0.392 (735) | 0.0131 (1342) | 0.354 | 0.043 | 0.0045 | 0.278 | 0.0037 |
| otc_token_word | 0.393 (736) | 0.0143 (1462) | 0.335 | 0.051 | 0.0073 | 0.273 | 0.0037 |
| otc_token_word_recip | 0.385 (722) | 0.0149 (1529) | 0.321 | 0.050 | 0.0055 | 0.334 | 0.0048 |
| otc_word | 0.309 (579) | 0.0109 (1115) | 0.342 | 0.031 | 0.0035 | 0.316 | 0.0055 |
| otc_word_tuned | 0.309 (580) | 0.0106 (1090) | 0.347 | 0.040 | 0.0066 | 0.267 | 0.0028 |

**Disputed chars with star > 0.5, by character class** (token probe, w=0)

| class | n disputed | ctc | otc_recip | otc_token | otc_token_word | otc_token_word_recip | otc_word | otc_word_tuned |
|---|---|---|---|---|---|---|---|---|
| consonant | 680 | 0.36 | 0.42 | 0.43 | 0.42 | 0.39 | 0.35 | 0.36 |
| vowel_sign | 678 | 0.30 | 0.44 | 0.43 | 0.43 | 0.43 | 0.32 | 0.32 |
| indep_vowel | 270 | 0.22 | 0.33 | 0.29 | 0.31 | 0.32 | 0.21 | 0.21 |
| space | 106 | 0.13 | 0.25 | 0.25 | 0.25 | 0.23 | 0.24 | 0.18 |
| other_mark | 76 | 0.30 | 0.42 | 0.33 | 0.37 | 0.38 | 0.33 | 0.32 |
| virama | 64 | 0.27 | 0.38 | 0.39 | 0.33 | 0.39 | 0.25 | 0.31 |

**vs CTC control** (adjusted OR of star on disputed units; ratio = OTC/CTC; dAUC with 300-sample cluster bootstrap)

| model | unit | outcome | OR CTC | OR model | ratio (model/CTC) | dAUC(p) | dAUC(-lam) |
|---|---|---|---|---|---|---|---|
| otc_recip | char | p_tok_w0>0.5 | 26.60 [22.76, 31.08] p=0.0e+00 | 27.85 [24.34, 31.87] p=0.0e+00 | 1.05 [0.93, 1.18] p=4.6e-01 | -0.004 [-0.014, +0.005] | -0.002 [-0.009, +0.005] |
| otc_recip | word | p_word_w0>0.5 | 4.66 [3.14, 6.92] p=2.3e-14 | 8.07 [5.87, 11.09] p=7.6e-38 | 1.73 [1.24, 2.42] p=1.3e-03 | +0.007 [+0.001, +0.013] | +0.009 [+0.001, +0.017] |
| otc_recip | bound | ins_w0>0.5 | 102.62 [78.38, 134.35] p=7.7e-249 | 66.78 [54.02, 82.56] p=0.0e+00 | 0.65 [0.53, 0.80] p=6.9e-05 | +0.005 [-0.020, +0.028] | - |
| otc_token | char | p_tok_w0>0.5 | 27.27 [23.34, 31.87] p=0.0e+00 | 32.78 [28.44, 37.79] p=0.0e+00 | 1.20 [1.06, 1.36] p=3.7e-03 | +0.018 [+0.009, +0.026] | +0.014 [+0.007, +0.022] |
| otc_token | word | p_word_w0>0.5 | 5.03 [3.43, 7.39] p=1.5e-16 | 7.61 [5.42, 10.68] p=7.5e-32 | 1.51 [1.13, 2.02] p=5.1e-03 | +0.001 [-0.004, +0.005] | -0.001 [-0.008, +0.005] |
| otc_token | bound | ins_w0>0.5 | 101.54 [77.62, 132.83] p=4.1e-249 | 76.37 [60.07, 97.10] p=2.7e-274 | 0.75 [0.61, 0.92] p=5.7e-03 | -0.003 [-0.020, +0.017] | - |
| otc_token_word | char | p_tok_w0>0.5 | 27.50 [23.53, 32.12] p=0.0e+00 | 31.38 [27.24, 36.17] p=0.0e+00 | 1.14 [1.01, 1.29] p=3.0e-02 | +0.018 [+0.010, +0.027] | +0.017 [+0.011, +0.025] |
| otc_token_word | word | p_word_w0>0.5 | 5.85 [4.05, 8.45] p=4.7e-21 | 5.16 [3.80, 7.01] p=6.0e-26 | 0.88 [0.65, 1.19] p=4.1e-01 | +0.007 [+0.001, +0.012] | +0.010 [+0.003, +0.017] |
| otc_token_word | bound | ins_w0>0.5 | 101.56 [77.54, 133.02] p=7.0e-247 | 72.69 [57.21, 92.36] p=1.4e-269 | 0.72 [0.58, 0.88] p=1.5e-03 | -0.009 [-0.026, +0.008] | - |
| otc_token_word_recip | char | p_tok_w0>0.5 | 26.89 [23.02, 31.43] p=0.0e+00 | 27.17 [23.67, 31.18] p=0.0e+00 | 1.01 [0.90, 1.14] p=8.7e-01 | -0.005 [-0.014, +0.003] | -0.005 [-0.013, +0.004] |
| otc_token_word_recip | word | p_word_w0>0.5 | 5.79 [3.99, 8.41] p=2.6e-20 | 7.53 [5.55, 10.22] p=1.6e-38 | 1.30 [0.98, 1.73] p=6.9e-02 | +0.006 [+0.001, +0.012] | +0.003 [-0.005, +0.011] |
| otc_token_word_recip | bound | ins_w0>0.5 | 101.99 [77.68, 133.91] p=6.0e-243 | 80.07 [64.54, 99.33] p=0.0e+00 | 0.79 [0.62, 0.99] p=3.8e-02 | -0.009 [-0.030, +0.013] | - |
| otc_word | char | p_tok_w0>0.5 | 28.91 [24.78, 33.72] p=0.0e+00 | 26.66 [22.90, 31.03] p=0.0e+00 | 0.92 [0.82, 1.04] p=1.9e-01 | -0.033 [-0.044, -0.024] | -0.021 [-0.031, -0.013] |
| otc_word | word | p_word_w0>0.5 | 5.25 [3.61, 7.63] p=4.0e-18 | 6.44 [4.40, 9.44] p=1.2e-21 | 1.23 [0.87, 1.73] p=2.4e-01 | +0.001 [-0.004, +0.006] | -0.001 [-0.008, +0.006] |
| otc_word | bound | ins_w0>0.5 | 107.03 [81.14, 141.17] p=5.9e-240 | 53.94 [42.85, 67.89] p=5.5e-253 | 0.50 [0.40, 0.63] p=2.2e-09 | +0.030 [+0.011, +0.047] | - |
| otc_word_tuned | char | p_tok_w0>0.5 | 29.20 [25.03, 34.06] p=0.0e+00 | 29.65 [25.47, 34.53] p=0.0e+00 | 1.02 [0.91, 1.13] p=7.8e-01 | -0.004 [-0.013, +0.005] | -0.002 [-0.011, +0.005] |
| otc_word_tuned | word | p_word_w0>0.5 | 5.80 [4.01, 8.39] p=1.1e-20 | 4.46 [3.22, 6.17] p=1.8e-19 | 0.77 [0.57, 1.03] p=7.9e-02 | +0.005 [+0.001, +0.009] | +0.012 [+0.006, +0.016] |
| otc_word_tuned | bound | ins_w0>0.5 | 104.64 [80.04, 136.81] p=1.9e-253 | 99.87 [77.07, 129.43] p=1.8e-265 | 0.95 [0.78, 1.16] p=6.5e-01 | -0.013 [-0.029, +0.002] | - |

## bn (113763 chars, 2683 disputed = 2.36%)

| model | unit | measure | mean(disputed) | mean(undisputed) | ratio | AUC |
|---|---|---|---|---|---|---|
| ctc | char | p_tok_w0 | 0.188 | 0.021 | 8.9 | 0.731 |
| ctc | char | neg_lam | -3.004 | -5.399 | - | 0.766 |
| ctc | word | p_word_w0 | 0.030 | 0.007 | 4.6 | 0.524 |
| ctc | word | neg_lam | -40.431 | -39.193 | - | 0.488 |
| ctc | bound | ins_w0 | 0.145 | 0.015 | 9.6 | 0.685 |
| otc_recip | char | p_tok_w0 | 0.253 | 0.023 | 11.2 | 0.741 |
| otc_recip | char | neg_lam | -2.247 | -7.171 | - | 0.756 |
| otc_recip | char | own_p | 0.243 | 0.020 | 12.3 | 0.739 |
| otc_recip | word | p_word_w0 | 0.049 | 0.008 | 6.3 | 0.536 |
| otc_recip | word | neg_lam | -47.672 | -46.757 | - | 0.492 |
| otc_recip | bound | ins_w0 | 0.208 | 0.026 | 8.0 | 0.667 |
| otc_token | char | p_tok_w0 | 0.228 | 0.022 | 10.5 | 0.760 |
| otc_token | char | neg_lam | -2.585 | -6.777 | - | 0.769 |
| otc_token | char | own_p | 0.221 | 0.020 | 10.9 | 0.759 |
| otc_token | word | p_word_w0 | 0.038 | 0.007 | 5.6 | 0.528 |
| otc_token | word | neg_lam | -46.397 | -44.897 | - | 0.487 |
| otc_token | bound | ins_w0 | 0.186 | 0.023 | 8.0 | 0.665 |
| otc_token_word | char | p_tok_w0 | 0.255 | 0.030 | 8.5 | 0.748 |
| otc_token_word | char | neg_lam | -2.078 | -6.656 | - | 0.758 |
| otc_token_word | char | own_p | 0.198 | 0.018 | 10.8 | 0.701 |
| otc_token_word | word | p_word_w0 | 0.084 | 0.023 | 3.6 | 0.539 |
| otc_token_word | word | neg_lam | -39.340 | -41.527 | - | 0.503 |
| otc_token_word | word | own_p | 0.055 | 0.019 | 2.8 | 0.526 |
| otc_token_word | bound | ins_w0 | 0.177 | 0.023 | 7.7 | 0.645 |
| otc_token_word_recip | char | p_tok_w0 | 0.237 | 0.023 | 10.4 | 0.730 |
| otc_token_word_recip | char | neg_lam | -2.431 | -7.050 | - | 0.746 |
| otc_token_word_recip | char | own_p | 0.201 | 0.018 | 11.0 | 0.706 |
| otc_token_word_recip | word | p_word_w0 | 0.055 | 0.009 | 5.8 | 0.534 |
| otc_token_word_recip | word | neg_lam | -45.962 | -44.905 | - | 0.486 |
| otc_token_word_recip | word | own_p | 0.018 | 0.004 | 4.6 | 0.515 |
| otc_token_word_recip | bound | ins_w0 | 0.195 | 0.025 | 7.7 | 0.677 |
| otc_word | char | p_tok_w0 | 0.201 | 0.022 | 9.2 | 0.706 |
| otc_word | char | neg_lam | -3.176 | -6.291 | - | 0.742 |
| otc_word | word | p_word_w0 | 0.034 | 0.008 | 4.2 | 0.522 |
| otc_word | word | neg_lam | -43.139 | -40.969 | - | 0.471 |
| otc_word | word | own_p | 0.048 | 0.010 | 4.9 | 0.533 |
| otc_word | bound | ins_w0 | 0.208 | 0.021 | 9.8 | 0.705 |
| otc_word_tuned | char | p_tok_w0 | 0.208 | 0.026 | 7.9 | 0.724 |
| otc_word_tuned | char | neg_lam | -2.673 | -5.451 | - | 0.760 |
| otc_word_tuned | word | p_word_w0 | 0.059 | 0.016 | 3.8 | 0.533 |
| otc_word_tuned | word | neg_lam | -38.871 | -40.249 | - | 0.507 |
| otc_word_tuned | word | own_p | 0.070 | 0.018 | 3.9 | 0.542 |
| otc_word_tuned | bound | ins_w0 | 0.175 | 0.016 | 11.1 | 0.680 |

**Share of units with star > 0.5** (disputed / undisputed; and how many of all starred units are disputed)

| model | char: disputed | char: undisputed | char: precision | word (p_word_w0): disputed | word: undisputed | boundary ins (w=0): disputed | boundary: undisputed |
|---|---|---|---|---|---|---|---|
| ctc | 0.175 (470) | 0.0108 (1199) | 0.282 | 0.030 | 0.0066 | 0.110 | 0.0025 |
| otc_recip | 0.248 (666) | 0.0172 (1912) | 0.258 | 0.049 | 0.0078 | 0.176 | 0.0062 |
| otc_token | 0.217 (582) | 0.0151 (1672) | 0.258 | 0.037 | 0.0067 | 0.135 | 0.0043 |
| otc_token_word | 0.243 (652) | 0.0231 (2565) | 0.203 | 0.084 | 0.0235 | 0.138 | 0.0046 |
| otc_token_word_recip | 0.231 (619) | 0.0173 (1917) | 0.244 | 0.053 | 0.0091 | 0.163 | 0.0059 |
| otc_word | 0.187 (503) | 0.0135 (1499) | 0.251 | 0.035 | 0.0083 | 0.165 | 0.0051 |
| otc_word_tuned | 0.196 (527) | 0.0162 (1800) | 0.226 | 0.059 | 0.0154 | 0.137 | 0.0029 |

**Disputed chars with star > 0.5, by character class** (token probe, w=0)

| class | n disputed | ctc | otc_recip | otc_token | otc_token_word | otc_token_word_recip | otc_word | otc_word_tuned |
|---|---|---|---|---|---|---|---|---|
| consonant | 1089 | 0.20 | 0.27 | 0.24 | 0.26 | 0.24 | 0.21 | 0.23 |
| vowel_sign | 800 | 0.16 | 0.23 | 0.20 | 0.21 | 0.22 | 0.17 | 0.17 |
| virama | 301 | 0.12 | 0.21 | 0.17 | 0.22 | 0.19 | 0.12 | 0.13 |
| indep_vowel | 273 | 0.21 | 0.30 | 0.27 | 0.31 | 0.30 | 0.24 | 0.24 |
| space | 185 | 0.10 | 0.17 | 0.14 | 0.17 | 0.16 | 0.12 | 0.11 |
| other_mark | 35 | 0.23 | 0.54 | 0.34 | 0.51 | 0.37 | 0.26 | 0.26 |

**vs CTC control** (adjusted OR of star on disputed units; ratio = OTC/CTC; dAUC with 300-sample cluster bootstrap)

| model | unit | outcome | OR CTC | OR model | ratio (model/CTC) | dAUC(p) | dAUC(-lam) |
|---|---|---|---|---|---|---|---|
| otc_recip | char | p_tok_w0>0.5 | 14.14 [12.13, 16.50] p=1.1e-249 | 14.50 [12.62, 16.66] p=0.0e+00 | 1.03 [0.91, 1.15] p=6.7e-01 | +0.009 [-0.002, +0.019] | -0.010 [-0.019, -0.000] |
| otc_recip | word | p_word_w0>0.5 | 4.14 [2.91, 5.89] p=3.2e-15 | 6.11 [4.53, 8.23] p=1.7e-32 | 1.48 [1.11, 1.96] p=6.9e-03 | +0.011 [+0.007, +0.016] | +0.004 [-0.005, +0.012] |
| otc_recip | bound | ins_w0>0.5 | 36.51 [27.99, 47.63] p=6.5e-155 | 32.89 [26.75, 40.45] p=2.7e-240 | 0.90 [0.73, 1.12] p=3.4e-01 | -0.018 [-0.035, +0.000] | - |
| otc_token | char | p_tok_w0>0.5 | 13.37 [11.43, 15.65] p=3.1e-229 | 14.33 [12.41, 16.55] p=7.8e-287 | 1.07 [0.96, 1.20] p=2.3e-01 | +0.028 [+0.019, +0.037] | +0.004 [-0.005, +0.013] |
| otc_token | word | p_word_w0>0.5 | 3.69 [2.55, 5.35] p=5.0e-12 | 5.42 [3.94, 7.44] p=2.2e-25 | 1.47 [1.11, 1.93] p=6.7e-03 | +0.003 [-0.001, +0.007] | -0.001 [-0.007, +0.006] |
| otc_token | bound | ins_w0>0.5 | 33.35 [25.32, 43.92] p=2.0e-137 | 35.86 [28.59, 44.98] p=2.1e-210 | 1.08 [0.88, 1.32] p=4.9e-01 | -0.020 [-0.036, -0.001] | - |
| otc_token_word | char | p_tok_w0>0.5 | 14.63 [12.54, 17.07] p=5.9e-255 | 12.77 [11.06, 14.74] p=2.0e-265 | 0.87 [0.77, 0.99] p=3.4e-02 | +0.016 [+0.007, +0.029] | -0.007 [-0.017, +0.001] |
| otc_token_word | word | p_word_w0>0.5 | 4.26 [3.00, 6.06] p=7.0e-16 | 3.10 [2.49, 3.85] p=2.1e-24 | 0.73 [0.54, 0.98] p=3.6e-02 | +0.015 [+0.010, +0.020] | +0.015 [+0.007, +0.023] |
| otc_token_word | bound | ins_w0>0.5 | 41.07 [31.37, 53.77] p=8.5e-161 | 33.32 [26.91, 41.25] p=4.1e-227 | 0.81 [0.65, 1.01] p=6.5e-02 | -0.040 [-0.060, -0.022] | - |
| otc_token_word_recip | char | p_tok_w0>0.5 | 14.14 [12.13, 16.49] p=2.6e-251 | 12.91 [11.13, 14.96] p=1.5e-252 | 0.91 [0.81, 1.03] p=1.4e-01 | -0.001 [-0.012, +0.009] | -0.020 [-0.030, -0.009] |
| otc_token_word_recip | word | p_word_w0>0.5 | 4.25 [2.99, 6.04] p=8.2e-16 | 5.49 [4.12, 7.31] p=2.8e-31 | 1.29 [0.97, 1.72] p=8.0e-02 | +0.010 [+0.005, +0.014] | -0.002 [-0.009, +0.005] |
| otc_token_word_recip | bound | ins_w0>0.5 | 36.03 [27.44, 47.31] p=9.5e-147 | 30.35 [24.42, 37.72] p=6.8e-208 | 0.84 [0.68, 1.04] p=1.1e-01 | -0.008 [-0.027, +0.010] | - |
| otc_word | char | p_tok_w0>0.5 | 13.65 [11.67, 15.96] p=2.5e-234 | 12.92 [11.00, 15.17] p=2.8e-214 | 0.95 [0.84, 1.07] p=3.6e-01 | -0.026 [-0.037, -0.014] | -0.024 [-0.034, -0.013] |
| otc_word | word | p_word_w0>0.5 | 3.69 [2.54, 5.36] p=8.5e-12 | 3.74 [2.70, 5.19] p=2.9e-15 | 1.01 [0.76, 1.36] p=9.2e-01 | -0.002 [-0.006, +0.002] | -0.017 [-0.023, -0.011] |
| otc_word | bound | ins_w0>0.5 | 36.49 [27.80, 47.89] p=4.4e-148 | 33.17 [26.82, 41.01] p=2.4e-229 | 0.91 [0.72, 1.15] p=4.2e-01 | +0.020 [+0.001, +0.039] | - |
| otc_word_tuned | char | p_tok_w0>0.5 | 13.70 [11.69, 16.04] p=1.3e-230 | 12.85 [11.05, 14.95] p=2.7e-240 | 0.94 [0.83, 1.06] p=3.0e-01 | -0.008 [-0.016, +0.001] | -0.005 [-0.013, +0.004] |
| otc_word_tuned | word | p_word_w0>0.5 | 4.24 [2.98, 6.03] p=9.1e-16 | 2.96 [2.30, 3.81] p=2.6e-17 | 0.70 [0.53, 0.93] p=1.4e-02 | +0.009 [+0.004, +0.014] | +0.018 [+0.013, +0.025] |
| otc_word_tuned | bound | ins_w0>0.5 | 39.86 [30.78, 51.61] p=8.7e-172 | 55.01 [43.33, 69.83] p=1.2e-237 | 1.38 [1.11, 1.71] p=3.2e-03 | -0.004 [-0.021, +0.011] | - |

## ch (120271 chars, 737 disputed = 0.61%)

| model | unit | measure | mean(disputed) | mean(undisputed) | ratio | AUC |
|---|---|---|---|---|---|---|
| ctc | char | p_tok_w0 | 0.235 | 0.018 | 13.4 | 0.781 |
| ctc | char | neg_lam | -2.619 | -5.784 | - | 0.799 |
| ctc | word | p_word_w0 | 0.015 | 0.003 | 4.9 | 0.521 |
| ctc | word | neg_lam | -38.303 | -37.639 | - | 0.509 |
| ctc | bound | ins_w0 | 0.121 | 0.015 | 8.1 | 0.696 |
| otc_recip | char | p_tok_w0 | 0.270 | 0.017 | 15.8 | 0.791 |
| otc_recip | char | neg_lam | -2.438 | -8.142 | - | 0.792 |
| otc_recip | char | own_p | 0.263 | 0.016 | 16.4 | 0.791 |
| otc_recip | word | p_word_w0 | 0.025 | 0.004 | 5.8 | 0.528 |
| otc_recip | word | neg_lam | -43.993 | -43.024 | - | 0.497 |
| otc_recip | bound | ins_w0 | 0.154 | 0.025 | 6.1 | 0.711 |
| otc_token | char | p_tok_w0 | 0.272 | 0.017 | 16.0 | 0.789 |
| otc_token | char | neg_lam | -2.341 | -8.072 | - | 0.794 |
| otc_token | char | own_p | 0.263 | 0.016 | 16.3 | 0.790 |
| otc_token | word | p_word_w0 | 0.029 | 0.004 | 7.6 | 0.530 |
| otc_token | word | neg_lam | -43.981 | -42.835 | - | 0.494 |
| otc_token | bound | ins_w0 | 0.149 | 0.025 | 6.0 | 0.733 |
| otc_token_word | char | p_tok_w0 | 0.260 | 0.017 | 15.5 | 0.803 |
| otc_token_word | char | neg_lam | -2.465 | -7.765 | - | 0.805 |
| otc_token_word | char | own_p | 0.235 | 0.015 | 15.7 | 0.790 |
| otc_token_word | word | p_word_w0 | 0.031 | 0.004 | 7.1 | 0.528 |
| otc_token_word | word | neg_lam | -42.404 | -41.182 | - | 0.494 |
| otc_token_word | word | own_p | 0.008 | 0.002 | 3.2 | 0.515 |
| otc_token_word | bound | ins_w0 | 0.162 | 0.024 | 6.8 | 0.724 |
| otc_token_word_recip | char | p_tok_w0 | 0.266 | 0.018 | 15.1 | 0.792 |
| otc_token_word_recip | char | neg_lam | -2.556 | -7.927 | - | 0.790 |
| otc_token_word_recip | char | own_p | 0.243 | 0.016 | 15.3 | 0.775 |
| otc_token_word_recip | word | p_word_w0 | 0.029 | 0.005 | 6.2 | 0.526 |
| otc_token_word_recip | word | neg_lam | -42.531 | -41.285 | - | 0.494 |
| otc_token_word_recip | word | own_p | 0.005 | 0.002 | 2.9 | 0.507 |
| otc_token_word_recip | bound | ins_w0 | 0.135 | 0.025 | 5.4 | 0.727 |
| otc_word | char | p_tok_w0 | 0.228 | 0.016 | 13.9 | 0.751 |
| otc_word | char | neg_lam | -3.348 | -6.923 | - | 0.773 |
| otc_word | word | p_word_w0 | 0.021 | 0.003 | 7.0 | 0.521 |
| otc_word | word | neg_lam | -37.305 | -34.740 | - | 0.483 |
| otc_word | word | own_p | 0.015 | 0.003 | 4.8 | 0.521 |
| otc_word | bound | ins_w0 | 0.145 | 0.021 | 6.8 | 0.767 |
| otc_word_tuned | char | p_tok_w0 | 0.225 | 0.017 | 13.2 | 0.792 |
| otc_word_tuned | char | neg_lam | -2.702 | -6.047 | - | 0.804 |
| otc_word_tuned | word | p_word_w0 | 0.017 | 0.003 | 5.1 | 0.522 |
| otc_word_tuned | word | neg_lam | -38.945 | -38.539 | - | 0.511 |
| otc_word_tuned | word | own_p | 0.027 | 0.004 | 6.3 | 0.537 |
| otc_word_tuned | bound | ins_w0 | 0.123 | 0.017 | 7.4 | 0.720 |

**Share of units with star > 0.5** (disputed / undisputed; and how many of all starred units are disputed)

| model | char: disputed | char: undisputed | char: precision | word (p_word_w0): disputed | word: undisputed | boundary ins (w=0): disputed | boundary: undisputed |
|---|---|---|---|---|---|---|---|
| ctc | 0.224 (165) | 0.0088 (1046) | 0.136 | 0.013 | 0.0027 | 0.082 | 0.0020 |
| otc_recip | 0.261 (192) | 0.0131 (1569) | 0.109 | 0.025 | 0.0042 | 0.101 | 0.0037 |
| otc_token | 0.265 (195) | 0.0129 (1543) | 0.112 | 0.030 | 0.0036 | 0.101 | 0.0035 |
| otc_token_word | 0.250 (184) | 0.0123 (1466) | 0.112 | 0.031 | 0.0043 | 0.101 | 0.0032 |
| otc_token_word_recip | 0.259 (191) | 0.0138 (1647) | 0.104 | 0.027 | 0.0047 | 0.091 | 0.0036 |
| otc_word | 0.216 (159) | 0.0094 (1125) | 0.124 | 0.020 | 0.0029 | 0.106 | 0.0042 |
| otc_word_tuned | 0.210 (155) | 0.0090 (1080) | 0.126 | 0.016 | 0.0033 | 0.086 | 0.0022 |

**Disputed chars with star > 0.5, by character class** (token probe, w=0)

| class | n disputed | ctc | otc_recip | otc_token | otc_token_word | otc_token_word_recip | otc_word | otc_word_tuned |
|---|---|---|---|---|---|---|---|---|
| consonant | 341 | 0.23 | 0.23 | 0.24 | 0.23 | 0.23 | 0.21 | 0.21 |
| vowel_sign | 235 | 0.19 | 0.27 | 0.28 | 0.27 | 0.28 | 0.20 | 0.20 |
| other_mark | 78 | 0.26 | 0.33 | 0.31 | 0.29 | 0.29 | 0.23 | 0.19 |
| indep_vowel | 52 | 0.25 | 0.27 | 0.25 | 0.27 | 0.25 | 0.27 | 0.25 |
| space | 24 | 0.21 | 0.21 | 0.25 | 0.17 | 0.25 | 0.21 | 0.21 |
| virama | 7 | 0.57 | 0.57 | 0.43 | 0.43 | 0.57 | 0.29 | 0.43 |

**vs CTC control** (adjusted OR of star on disputed units; ratio = OTC/CTC; dAUC with 300-sample cluster bootstrap)

| model | unit | outcome | OR CTC | OR model | ratio (model/CTC) | dAUC(p) | dAUC(-lam) |
|---|---|---|---|---|---|---|---|
| otc_recip | char | p_tok_w0>0.5 | 16.00 [11.70, 21.89] p=2.3e-67 | 17.93 [14.09, 22.82] p=1.6e-121 | 1.12 [0.93, 1.35] p=2.3e-01 | +0.010 [-0.009, +0.026] | -0.007 [-0.024, +0.009] |
| otc_recip | word | p_word_w0>0.5 | 2.97 [1.35, 6.54] p=6.7e-03 | 6.44 [4.11, 10.06] p=3.4e-16 | 2.16 [1.14, 4.12] p=1.9e-02 | +0.007 [+0.001, +0.013] | -0.012 [-0.022, -0.002] |
| otc_recip | bound | ins_w0>0.5 | 12.57 [5.04, 31.34] p=5.6e-08 | 39.96 [29.83, 53.52] p=5.8e-135 | 3.18 [1.49, 6.79] p=2.8e-03 | +0.015 [-0.012, +0.044] | - |
| otc_token | char | p_tok_w0>0.5 | 17.46 [12.86, 23.71] p=5.0e-75 | 20.00 [15.83, 25.27] p=4.5e-139 | 1.15 [0.94, 1.39] p=1.7e-01 | +0.007 [-0.010, +0.023] | -0.006 [-0.024, +0.011] |
| otc_token | word | p_word_w0>0.5 | 3.34 [1.61, 6.93] p=1.2e-03 | 8.94 [5.70, 14.02] p=1.4e-21 | 2.67 [1.44, 4.96] p=1.8e-03 | +0.008 [+0.002, +0.016] | -0.015 [-0.025, -0.007] |
| otc_token | bound | ins_w0>0.5 | 12.48 [4.92, 31.67] p=1.1e-07 | 39.84 [29.77, 53.31] p=9.3e-136 | 3.19 [1.46, 6.98] p=3.7e-03 | +0.037 [+0.011, +0.066] | - |
| otc_token_word | char | p_tok_w0>0.5 | 16.42 [12.03, 22.41] p=1.0e-69 | 19.17 [15.01, 24.47] p=4.9e-124 | 1.17 [0.96, 1.42] p=1.2e-01 | +0.022 [+0.006, +0.037] | +0.006 [-0.011, +0.020] |
| otc_token_word | word | p_word_w0>0.5 | 3.15 [1.47, 6.73] p=3.1e-03 | 7.74 [4.97, 12.05] p=1.3e-19 | 2.46 [1.34, 4.53] p=3.8e-03 | +0.007 [+0.001, +0.013] | -0.015 [-0.023, -0.006] |
| otc_token_word | bound | ins_w0>0.5 | 14.12 [6.22, 32.06] p=2.5e-10 | 45.67 [33.66, 61.97] p=5.5e-133 | 3.24 [1.72, 6.08] p=2.6e-04 | +0.029 [+0.001, +0.056] | - |
| otc_token_word_recip | char | p_tok_w0>0.5 | 15.97 [11.66, 21.87] p=9.5e-67 | 17.41 [13.72, 22.08] p=1.6e-122 | 1.09 [0.89, 1.34] p=4.1e-01 | +0.011 [-0.005, +0.028] | -0.009 [-0.026, +0.006] |
| otc_token_word_recip | word | p_word_w0>0.5 | 2.86 [1.27, 6.47] p=1.1e-02 | 6.10 [3.91, 9.54] p=2.0e-15 | 2.13 [1.11, 4.10] p=2.3e-02 | +0.005 [-0.002, +0.012] | -0.015 [-0.024, -0.007] |
| otc_token_word_recip | bound | ins_w0>0.5 | 13.12 [5.59, 30.81] p=3.4e-09 | 38.77 [28.74, 52.31] p=1.0e-126 | 2.96 [1.47, 5.93] p=2.3e-03 | +0.031 [+0.003, +0.054] | - |
| otc_word | char | p_tok_w0>0.5 | 18.63 [13.86, 25.05] p=1.7e-83 | 20.64 [15.99, 26.65] p=1.5e-119 | 1.11 [0.93, 1.33] p=2.6e-01 | -0.030 [-0.050, -0.013] | -0.026 [-0.044, -0.006] |
| otc_word | word | p_word_w0>0.5 | 3.29 [1.55, 6.97] p=1.9e-03 | 7.49 [4.55, 12.35] p=2.7e-15 | 2.28 [1.25, 4.16] p=7.4e-03 | -0.000 [-0.007, +0.006] | -0.026 [-0.033, -0.016] |
| otc_word | bound | ins_w0>0.5 | 8.08 [1.99, 32.85] p=3.5e-03 | 22.64 [16.60, 30.87] p=1.3e-86 | 2.80 [0.80, 9.87] p=1.1e-01 | +0.071 [+0.047, +0.096] | - |
| otc_word_tuned | char | p_tok_w0>0.5 | 18.56 [13.80, 24.95] p=2.5e-83 | 20.61 [15.77, 26.94] p=9.2e-109 | 1.11 [0.95, 1.29] p=1.8e-01 | +0.011 [-0.004, +0.025] | +0.005 [-0.010, +0.019] |
| otc_word_tuned | word | p_word_w0>0.5 | 2.91 [1.28, 6.65] p=1.1e-02 | 5.62 [3.24, 9.75] p=8.6e-10 | 1.93 [1.14, 3.26] p=1.4e-02 | +0.001 [-0.005, +0.006] | +0.002 [-0.004, +0.007] |
| otc_word_tuned | bound | ins_w0>0.5 | 16.05 [7.54, 34.18] p=6.2e-13 | 58.73 [43.12, 79.99] p=3.4e-147 | 3.66 [2.01, 6.66] p=2.2e-05 | +0.024 [+0.002, +0.049] | - |

## hi (121181 chars, 1230 disputed = 1.02%)

| model | unit | measure | mean(disputed) | mean(undisputed) | ratio | AUC |
|---|---|---|---|---|---|---|
| ctc | char | p_tok_w0 | 0.241 | 0.022 | 11.2 | 0.734 |
| ctc | char | neg_lam | -2.716 | -5.585 | - | 0.765 |
| ctc | word | p_word_w0 | 0.020 | 0.005 | 4.1 | 0.519 |
| ctc | word | neg_lam | -45.621 | -36.826 | - | 0.403 |
| ctc | bound | ins_w0 | 0.237 | 0.014 | 16.9 | 0.728 |
| otc_recip | char | p_tok_w0 | 0.302 | 0.022 | 13.5 | 0.764 |
| otc_recip | char | neg_lam | -1.746 | -7.479 | - | 0.778 |
| otc_recip | char | own_p | 0.290 | 0.020 | 14.6 | 0.766 |
| otc_recip | word | p_word_w0 | 0.044 | 0.007 | 6.3 | 0.529 |
| otc_recip | word | neg_lam | -54.843 | -42.463 | - | 0.376 |
| otc_recip | bound | ins_w0 | 0.277 | 0.023 | 11.8 | 0.726 |
| otc_token | char | p_tok_w0 | 0.298 | 0.022 | 13.5 | 0.767 |
| otc_token | char | neg_lam | -1.594 | -7.408 | - | 0.780 |
| otc_token | char | own_p | 0.291 | 0.021 | 14.0 | 0.767 |
| otc_token | word | p_word_w0 | 0.047 | 0.007 | 7.2 | 0.528 |
| otc_token | word | neg_lam | -54.438 | -42.234 | - | 0.373 |
| otc_token | bound | ins_w0 | 0.263 | 0.023 | 11.5 | 0.734 |
| otc_token_word | char | p_tok_w0 | 0.295 | 0.025 | 11.8 | 0.769 |
| otc_token_word | char | neg_lam | -1.850 | -7.039 | - | 0.790 |
| otc_token_word | char | own_p | 0.230 | 0.018 | 12.6 | 0.689 |
| otc_token_word | word | p_word_w0 | 0.074 | 0.014 | 5.4 | 0.544 |
| otc_token_word | word | neg_lam | -42.420 | -37.352 | - | 0.433 |
| otc_token_word | word | own_p | 0.048 | 0.010 | 4.8 | 0.531 |
| otc_token_word | bound | ins_w0 | 0.224 | 0.021 | 10.5 | 0.722 |
| otc_token_word_recip | char | p_tok_w0 | 0.305 | 0.023 | 13.5 | 0.753 |
| otc_token_word_recip | char | neg_lam | -1.728 | -7.298 | - | 0.766 |
| otc_token_word_recip | char | own_p | 0.275 | 0.018 | 15.0 | 0.726 |
| otc_token_word_recip | word | p_word_w0 | 0.048 | 0.007 | 6.6 | 0.534 |
| otc_token_word_recip | word | neg_lam | -48.418 | -39.126 | - | 0.404 |
| otc_token_word_recip | word | own_p | 0.012 | 0.003 | 4.3 | 0.513 |
| otc_token_word_recip | bound | ins_w0 | 0.266 | 0.023 | 11.7 | 0.728 |
| otc_word | char | p_tok_w0 | 0.258 | 0.021 | 12.3 | 0.730 |
| otc_word | char | neg_lam | -2.666 | -6.503 | - | 0.765 |
| otc_word | word | p_word_w0 | 0.026 | 0.006 | 4.6 | 0.522 |
| otc_word | word | neg_lam | -44.374 | -34.163 | - | 0.393 |
| otc_word | word | own_p | 0.047 | 0.008 | 6.2 | 0.539 |
| otc_word | bound | ins_w0 | 0.356 | 0.019 | 18.8 | 0.775 |
| otc_word_tuned | char | p_tok_w0 | 0.288 | 0.028 | 10.2 | 0.747 |
| otc_word_tuned | char | neg_lam | -2.181 | -6.069 | - | 0.771 |
| otc_word_tuned | word | p_word_w0 | 0.091 | 0.019 | 4.9 | 0.551 |
| otc_word_tuned | word | neg_lam | -38.695 | -36.143 | - | 0.456 |
| otc_word_tuned | word | own_p | 0.103 | 0.021 | 4.9 | 0.565 |
| otc_word_tuned | bound | ins_w0 | 0.225 | 0.016 | 13.6 | 0.709 |

**Share of units with star > 0.5** (disputed / undisputed; and how many of all starred units are disputed)

| model | char: disputed | char: undisputed | char: precision | word (p_word_w0): disputed | word: undisputed | boundary ins (w=0): disputed | boundary: undisputed |
|---|---|---|---|---|---|---|---|
| ctc | 0.229 (282) | 0.0126 (1516) | 0.157 | 0.021 | 0.0048 | 0.163 | 0.0026 |
| otc_recip | 0.298 (367) | 0.0177 (2129) | 0.147 | 0.045 | 0.0068 | 0.236 | 0.0052 |
| otc_token | 0.293 (361) | 0.0172 (2069) | 0.149 | 0.048 | 0.0065 | 0.210 | 0.0050 |
| otc_token_word | 0.290 (357) | 0.0195 (2339) | 0.132 | 0.074 | 0.0138 | 0.174 | 0.0042 |
| otc_token_word_recip | 0.300 (369) | 0.0178 (2131) | 0.148 | 0.048 | 0.0071 | 0.207 | 0.0048 |
| otc_word | 0.253 (311) | 0.0136 (1626) | 0.161 | 0.026 | 0.0056 | 0.293 | 0.0047 |
| otc_word_tuned | 0.282 (347) | 0.0204 (2450) | 0.124 | 0.091 | 0.0187 | 0.174 | 0.0031 |

**Disputed chars with star > 0.5, by character class** (token probe, w=0)

| class | n disputed | ctc | otc_recip | otc_token | otc_token_word | otc_token_word_recip | otc_word | otc_word_tuned |
|---|---|---|---|---|---|---|---|---|
| consonant | 504 | 0.25 | 0.29 | 0.27 | 0.30 | 0.29 | 0.27 | 0.28 |
| vowel_sign | 312 | 0.29 | 0.37 | 0.38 | 0.35 | 0.37 | 0.32 | 0.34 |
| other_mark | 163 | 0.11 | 0.25 | 0.25 | 0.23 | 0.28 | 0.13 | 0.22 |
| virama | 147 | 0.15 | 0.26 | 0.24 | 0.24 | 0.24 | 0.20 | 0.24 |
| indep_vowel | 71 | 0.30 | 0.32 | 0.30 | 0.31 | 0.32 | 0.30 | 0.30 |
| space | 33 | 0.15 | 0.18 | 0.21 | 0.18 | 0.15 | 0.15 | 0.18 |

**vs CTC control** (adjusted OR of star on disputed units; ratio = OTC/CTC; dAUC with 300-sample cluster bootstrap)

| model | unit | outcome | OR CTC | OR model | ratio (model/CTC) | dAUC(p) | dAUC(-lam) |
|---|---|---|---|---|---|---|---|
| otc_recip | char | p_tok_w0>0.5 | 11.31 [9.02, 14.17] p=2.0e-98 | 11.68 [9.47, 14.40] p=5.5e-117 | 1.03 [0.89, 1.20] p=6.7e-01 | +0.030 [+0.016, +0.045] | +0.013 [+0.001, +0.027] |
| otc_recip | word | p_word_w0>0.5 | 2.72 [1.53, 4.84] p=6.4e-04 | 6.04 [4.25, 8.57] p=8.5e-24 | 2.22 [1.30, 3.79] p=3.7e-03 | +0.010 [+0.004, +0.015] | -0.028 [-0.040, -0.015] |
| otc_recip | bound | ins_w0>0.5 | 52.20 [36.09, 75.49] p=5.9e-98 | 39.27 [29.24, 52.74] p=2.1e-131 | 0.75 [0.57, 0.99] p=4.5e-02 | -0.001 [-0.026, +0.025] | - |
| otc_token | char | p_tok_w0>0.5 | 11.32 [9.03, 14.19] p=2.0e-98 | 11.79 [9.62, 14.44] p=1.0e-125 | 1.04 [0.90, 1.21] p=6.0e-01 | +0.033 [+0.020, +0.047] | +0.015 [+0.001, +0.027] |
| otc_token | word | p_word_w0>0.5 | 2.90 [1.66, 5.08] p=1.8e-04 | 7.11 [4.86, 10.41] p=5.2e-24 | 2.45 [1.49, 4.03] p=4.2e-04 | +0.009 [+0.003, +0.014] | -0.031 [-0.042, -0.017] |
| otc_token | bound | ins_w0>0.5 | 44.21 [29.61, 66.01] p=1.3e-76 | 37.28 [27.51, 50.50] p=1.1e-120 | 0.84 [0.64, 1.10] p=2.2e-01 | +0.006 [-0.018, +0.036] | - |
| otc_token_word | char | p_tok_w0>0.5 | 11.41 [9.11, 14.31] p=5.4e-99 | 11.41 [9.09, 14.33] p=1.1e-97 | 1.00 [0.85, 1.18] p=1.0e+00 | +0.035 [+0.022, +0.047] | +0.025 [+0.011, +0.038] |
| otc_token_word | word | p_word_w0>0.5 | 2.29 [1.18, 4.45] p=1.5e-02 | 3.73 [2.80, 4.97] p=2.2e-19 | 1.63 [0.91, 2.92] p=1.0e-01 | +0.024 [+0.017, +0.033] | +0.030 [+0.020, +0.042] |
| otc_token_word | bound | ins_w0>0.5 | 47.61 [32.02, 70.79] p=3.4e-81 | 40.08 [29.06, 55.30] p=6.1e-112 | 0.84 [0.63, 1.12] p=2.4e-01 | -0.006 [-0.036, +0.025] | - |
| otc_token_word_recip | char | p_tok_w0>0.5 | 11.15 [8.89, 13.98] p=8.6e-97 | 11.79 [9.61, 14.46] p=1.0e-123 | 1.06 [0.91, 1.22] p=4.5e-01 | +0.019 [+0.003, +0.034] | +0.001 [-0.014, +0.014] |
| otc_token_word_recip | word | p_word_w0>0.5 | 2.85 [1.60, 5.08] p=3.9e-04 | 5.72 [3.94, 8.31] p=5.8e-20 | 2.01 [1.22, 3.31] p=6.1e-03 | +0.014 [+0.009, +0.021] | +0.000 [-0.010, +0.010] |
| otc_token_word_recip | bound | ins_w0>0.5 | 46.98 [31.85, 69.29] p=5.4e-84 | 43.88 [32.64, 59.00] p=2.2e-138 | 0.93 [0.74, 1.19] p=5.7e-01 | +0.000 [-0.030, +0.025] | - |
| otc_word | char | p_tok_w0>0.5 | 11.61 [9.28, 14.53] p=4.3e-102 | 13.25 [10.67, 16.47] p=1.3e-120 | 1.14 [0.99, 1.32] p=7.8e-02 | -0.004 [-0.019, +0.011] | +0.000 [-0.014, +0.012] |
| otc_word | word | p_word_w0>0.5 | 2.59 [1.40, 4.77] p=2.3e-03 | 4.45 [2.82, 7.01] p=1.3e-10 | 1.72 [1.00, 2.96] p=5.1e-02 | +0.002 [-0.004, +0.008] | -0.011 [-0.021, -0.001] |
| otc_word | bound | ins_w0>0.5 | 37.70 [24.78, 57.37] p=1.9e-64 | 53.99 [41.24, 70.68] p=3.4e-185 | 1.43 [1.05, 1.96] p=2.4e-02 | +0.047 [+0.022, +0.071] | - |
| otc_word_tuned | char | p_tok_w0>0.5 | 11.61 [9.27, 14.53] p=2.7e-101 | 10.43 [8.39, 12.97] p=7.6e-99 | 0.90 [0.77, 1.05] p=1.7e-01 | +0.013 [-0.003, +0.027] | +0.006 [-0.008, +0.019] |
| otc_word_tuned | word | p_word_w0>0.5 | 3.81 [2.26, 6.40] p=4.6e-07 | 2.84 [2.17, 3.70] p=1.6e-14 | 0.75 [0.46, 1.20] p=2.3e-01 | +0.031 [+0.024, +0.041] | +0.052 [+0.040, +0.062] |
| otc_word_tuned | bound | ins_w0>0.5 | 24.23 [14.25, 41.20] p=5.4e-32 | 76.08 [56.06, 103.27] p=4.9e-170 | 3.14 [2.08, 4.74] p=5.0e-08 | -0.019 [-0.053, +0.012] | - |

## kn (126258 chars, 1729 disputed = 1.37%)

| model | unit | measure | mean(disputed) | mean(undisputed) | ratio | AUC |
|---|---|---|---|---|---|---|
| ctc | char | p_tok_w0 | 0.252 | 0.021 | 12.2 | 0.815 |
| ctc | char | neg_lam | -2.397 | -5.260 | - | 0.838 |
| ctc | word | p_word_w0 | 0.011 | 0.005 | 2.4 | 0.514 |
| ctc | word | neg_lam | -54.223 | -53.658 | - | 0.497 |
| ctc | bound | ins_w0 | 0.204 | 0.014 | 15.0 | 0.807 |
| otc_recip | char | p_tok_w0 | 0.335 | 0.023 | 14.6 | 0.823 |
| otc_recip | char | neg_lam | -1.255 | -6.956 | - | 0.842 |
| otc_recip | char | own_p | 0.322 | 0.020 | 16.1 | 0.824 |
| otc_recip | word | p_word_w0 | 0.018 | 0.006 | 3.1 | 0.513 |
| otc_recip | word | neg_lam | -61.547 | -59.702 | - | 0.489 |
| otc_recip | bound | ins_w0 | 0.298 | 0.024 | 12.4 | 0.786 |
| otc_token | char | p_tok_w0 | 0.322 | 0.022 | 14.9 | 0.832 |
| otc_token | char | neg_lam | -1.389 | -6.790 | - | 0.855 |
| otc_token | char | own_p | 0.315 | 0.020 | 15.5 | 0.833 |
| otc_token | word | p_word_w0 | 0.021 | 0.006 | 3.4 | 0.512 |
| otc_token | word | neg_lam | -61.072 | -59.386 | - | 0.491 |
| otc_token | bound | ins_w0 | 0.277 | 0.022 | 12.4 | 0.793 |
| otc_token_word | char | p_tok_w0 | 0.319 | 0.025 | 12.9 | 0.834 |
| otc_token_word | char | neg_lam | -1.621 | -6.225 | - | 0.849 |
| otc_token_word | char | own_p | 0.293 | 0.019 | 15.3 | 0.823 |
| otc_token_word | word | p_word_w0 | 0.032 | 0.015 | 2.1 | 0.517 |
| otc_token_word | word | neg_lam | -53.901 | -53.797 | - | 0.498 |
| otc_token_word | word | own_p | 0.023 | 0.013 | 1.8 | 0.514 |
| otc_token_word | bound | ins_w0 | 0.246 | 0.020 | 12.3 | 0.784 |
| otc_token_word_recip | char | p_tok_w0 | 0.335 | 0.024 | 14.2 | 0.822 |
| otc_token_word_recip | char | neg_lam | -1.337 | -6.892 | - | 0.839 |
| otc_token_word_recip | char | own_p | 0.306 | 0.019 | 16.3 | 0.816 |
| otc_token_word_recip | word | p_word_w0 | 0.024 | 0.009 | 2.7 | 0.515 |
| otc_token_word_recip | word | neg_lam | -60.183 | -58.715 | - | 0.493 |
| otc_token_word_recip | word | own_p | 0.008 | 0.004 | 2.0 | 0.508 |
| otc_token_word_recip | bound | ins_w0 | 0.289 | 0.024 | 12.0 | 0.783 |
| otc_word | char | p_tok_w0 | 0.266 | 0.021 | 13.0 | 0.792 |
| otc_word | char | neg_lam | -2.615 | -6.097 | - | 0.813 |
| otc_word | word | p_word_w0 | 0.014 | 0.005 | 2.9 | 0.511 |
| otc_word | word | neg_lam | -55.031 | -53.491 | - | 0.490 |
| otc_word | word | own_p | 0.015 | 0.005 | 3.1 | 0.515 |
| otc_word | bound | ins_w0 | 0.283 | 0.020 | 14.5 | 0.845 |
| otc_word_tuned | char | p_tok_w0 | 0.242 | 0.025 | 9.5 | 0.812 |
| otc_word_tuned | char | neg_lam | -2.286 | -5.059 | - | 0.838 |
| otc_word_tuned | word | p_word_w0 | 0.039 | 0.017 | 2.3 | 0.522 |
| otc_word_tuned | word | neg_lam | -49.613 | -51.036 | - | 0.512 |
| otc_word_tuned | word | own_p | 0.045 | 0.019 | 2.4 | 0.528 |
| otc_word_tuned | bound | ins_w0 | 0.212 | 0.013 | 16.4 | 0.795 |

**Share of units with star > 0.5** (disputed / undisputed; and how many of all starred units are disputed)

| model | char: disputed | char: undisputed | char: precision | word (p_word_w0): disputed | word: undisputed | boundary ins (w=0): disputed | boundary: undisputed |
|---|---|---|---|---|---|---|---|
| ctc | 0.236 (408) | 0.0098 (1222) | 0.250 | 0.010 | 0.0045 | 0.167 | 0.0023 |
| otc_recip | 0.320 (554) | 0.0168 (2088) | 0.210 | 0.018 | 0.0057 | 0.266 | 0.0053 |
| otc_token | 0.308 (532) | 0.0150 (1874) | 0.221 | 0.021 | 0.0059 | 0.245 | 0.0044 |
| otc_token_word | 0.312 (539) | 0.0168 (2090) | 0.205 | 0.032 | 0.0149 | 0.207 | 0.0035 |
| otc_token_word_recip | 0.327 (565) | 0.0174 (2161) | 0.207 | 0.025 | 0.0085 | 0.253 | 0.0054 |
| otc_word | 0.248 (429) | 0.0113 (1409) | 0.233 | 0.014 | 0.0046 | 0.236 | 0.0048 |
| otc_word_tuned | 0.218 (377) | 0.0141 (1761) | 0.176 | 0.038 | 0.0176 | 0.164 | 0.0025 |

**Disputed chars with star > 0.5, by character class** (token probe, w=0)

| class | n disputed | ctc | otc_recip | otc_token | otc_token_word | otc_token_word_recip | otc_word | otc_word_tuned |
|---|---|---|---|---|---|---|---|---|
| vowel_sign | 577 | 0.30 | 0.39 | 0.38 | 0.38 | 0.41 | 0.31 | 0.27 |
| consonant | 501 | 0.21 | 0.25 | 0.23 | 0.26 | 0.25 | 0.21 | 0.20 |
| virama | 380 | 0.16 | 0.26 | 0.26 | 0.26 | 0.25 | 0.19 | 0.15 |
| space | 202 | 0.18 | 0.33 | 0.31 | 0.27 | 0.37 | 0.25 | 0.18 |
| indep_vowel | 64 | 0.41 | 0.53 | 0.53 | 0.52 | 0.52 | 0.31 | 0.41 |
| other_mark | 5 | 0.40 | 0.60 | 0.60 | 0.60 | 0.60 | 0.60 | 0.40 |

**vs CTC control** (adjusted OR of star on disputed units; ratio = OTC/CTC; dAUC with 300-sample cluster bootstrap)

| model | unit | outcome | OR CTC | OR model | ratio (model/CTC) | dAUC(p) | dAUC(-lam) |
|---|---|---|---|---|---|---|---|
| otc_recip | char | p_tok_w0>0.5 | 22.17 [18.91, 25.99] p=0.0e+00 | 19.40 [16.83, 22.37] p=0.0e+00 | 0.88 [0.78, 0.99] p=3.1e-02 | +0.008 [-0.001, +0.017] | +0.003 [-0.006, +0.011] |
| otc_recip | word | p_word_w0>0.5 | 1.41 [0.82, 2.41] p=2.1e-01 | 2.50 [1.67, 3.74] p=8.2e-06 | 1.78 [1.13, 2.79] p=1.3e-02 | -0.001 [-0.005, +0.002] | -0.008 [-0.016, -0.001] |
| otc_recip | bound | ins_w0>0.5 | 50.43 [39.49, 64.41] p=9.6e-217 | 45.08 [37.53, 54.16] p=0.0e+00 | 0.89 [0.73, 1.09] p=2.8e-01 | -0.022 [-0.035, -0.008] | - |
| otc_token | char | p_tok_w0>0.5 | 22.00 [18.78, 25.78] p=0.0e+00 | 19.76 [17.13, 22.79] p=0.0e+00 | 0.90 [0.80, 1.01] p=8.1e-02 | +0.017 [+0.008, +0.026] | +0.016 [+0.008, +0.025] |
| otc_token | word | p_word_w0>0.5 | 1.41 [0.83, 2.42] p=2.1e-01 | 2.75 [1.88, 4.03] p=1.9e-07 | 1.94 [1.26, 3.00] p=2.6e-03 | -0.002 [-0.005, +0.001] | -0.005 [-0.012, +0.001] |
| otc_token | bound | ins_w0>0.5 | 49.67 [39.15, 63.03] p=1.2e-226 | 47.48 [39.41, 57.19] p=0.0e+00 | 0.96 [0.80, 1.14] p=6.1e-01 | -0.015 [-0.027, -0.004] | - |
| otc_token_word | char | p_tok_w0>0.5 | 22.39 [19.11, 26.23] p=0.0e+00 | 21.50 [18.75, 24.66] p=0.0e+00 | 0.96 [0.85, 1.08] p=5.0e-01 | +0.020 [+0.011, +0.029] | +0.011 [+0.002, +0.019] |
| otc_token_word | word | p_word_w0>0.5 | 1.65 [1.00, 2.74] p=5.1e-02 | 1.47 [1.08, 1.99] p=1.4e-02 | 0.89 [0.55, 1.42] p=6.1e-01 | +0.003 [-0.002, +0.007] | +0.002 [-0.003, +0.008] |
| otc_token_word | bound | ins_w0>0.5 | 48.26 [37.82, 61.58] p=1.9e-213 | 50.98 [41.77, 62.22] p=0.0e+00 | 1.06 [0.88, 1.26] p=5.5e-01 | -0.023 [-0.039, -0.009] | - |
| otc_token_word_recip | char | p_tok_w0>0.5 | 22.06 [18.82, 25.86] p=0.0e+00 | 18.83 [16.36, 21.66] p=0.0e+00 | 0.85 [0.75, 0.97] p=1.2e-02 | +0.007 [-0.002, +0.017] | +0.000 [-0.008, +0.009] |
| otc_token_word_recip | word | p_word_w0>0.5 | 1.35 [0.77, 2.36] p=3.0e-01 | 2.17 [1.55, 3.04] p=6.2e-06 | 1.61 [0.98, 2.65] p=6.0e-02 | +0.001 [-0.003, +0.005] | -0.004 [-0.010, +0.004] |
| otc_token_word_recip | bound | ins_w0>0.5 | 49.63 [38.90, 63.31] p=8.6e-217 | 37.99 [31.46, 45.88] p=0.0e+00 | 0.77 [0.63, 0.93] p=6.5e-03 | -0.024 [-0.038, -0.011] | - |
| otc_word | char | p_tok_w0>0.5 | 21.53 [18.33, 25.30] p=2.2e-305 | 20.07 [17.13, 23.51] p=2.7e-301 | 0.93 [0.83, 1.05] p=2.5e-01 | -0.023 [-0.034, -0.011] | -0.026 [-0.037, -0.014] |
| otc_word | word | p_word_w0>0.5 | 1.46 [0.87, 2.45] p=1.5e-01 | 2.38 [1.50, 3.78] p=2.3e-04 | 1.63 [1.04, 2.56] p=3.3e-02 | -0.003 [-0.005, -0.000] | -0.007 [-0.011, -0.002] |
| otc_word | bound | ins_w0>0.5 | 39.38 [30.69, 50.54] p=2.4e-183 | 37.84 [30.81, 46.48] p=6.4e-263 | 0.96 [0.79, 1.17] p=6.9e-01 | +0.037 [+0.026, +0.050] | - |
| otc_word_tuned | char | p_tok_w0>0.5 | 22.68 [19.36, 26.57] p=0.0e+00 | 15.76 [13.42, 18.51] p=1.5e-247 | 0.69 [0.61, 0.79] p=1.0e-08 | -0.003 [-0.012, +0.006] | -0.000 [-0.009, +0.010] |
| otc_word_tuned | word | p_word_w0>0.5 | 1.30 [0.74, 2.28] p=3.7e-01 | 1.57 [1.19, 2.08] p=1.6e-03 | 1.21 [0.74, 2.00] p=4.5e-01 | +0.008 [+0.004, +0.013] | +0.015 [+0.011, +0.019] |
| otc_word_tuned | bound | ins_w0>0.5 | 47.49 [37.27, 60.51] p=6.1e-214 | 58.64 [46.50, 73.96] p=3.1e-259 | 1.23 [1.03, 1.48] p=2.1e-02 | -0.012 [-0.026, +0.001] | - |

## mg (104755 chars, 3748 disputed = 3.58%)

| model | unit | measure | mean(disputed) | mean(undisputed) | ratio | AUC |
|---|---|---|---|---|---|---|
| ctc | char | p_tok_w0 | 0.190 | 0.022 | 8.6 | 0.832 |
| ctc | char | neg_lam | -2.768 | -5.443 | - | 0.850 |
| ctc | word | p_word_w0 | 0.025 | 0.007 | 3.4 | 0.534 |
| ctc | word | neg_lam | -29.119 | -33.746 | - | 0.590 |
| ctc | bound | ins_w0 | 0.225 | 0.014 | 15.6 | 0.738 |
| otc_recip | char | p_tok_w0 | 0.295 | 0.026 | 11.3 | 0.821 |
| otc_recip | char | neg_lam | -1.810 | -7.265 | - | 0.834 |
| otc_recip | char | own_p | 0.281 | 0.022 | 12.5 | 0.825 |
| otc_recip | word | p_word_w0 | 0.039 | 0.008 | 5.0 | 0.531 |
| otc_recip | word | neg_lam | -36.913 | -41.229 | - | 0.578 |
| otc_recip | bound | ins_w0 | 0.312 | 0.025 | 12.4 | 0.745 |
| otc_token | char | p_tok_w0 | 0.263 | 0.024 | 10.9 | 0.841 |
| otc_token | char | neg_lam | -1.997 | -6.763 | - | 0.852 |
| otc_token | char | own_p | 0.257 | 0.023 | 11.4 | 0.844 |
| otc_token | word | p_word_w0 | 0.036 | 0.008 | 4.7 | 0.529 |
| otc_token | word | neg_lam | -35.093 | -39.646 | - | 0.580 |
| otc_token | bound | ins_w0 | 0.257 | 0.021 | 12.2 | 0.743 |
| otc_token_word | char | p_tok_w0 | 0.247 | 0.025 | 9.9 | 0.843 |
| otc_token_word | char | neg_lam | -2.137 | -6.511 | - | 0.855 |
| otc_token_word | char | own_p | 0.216 | 0.020 | 11.0 | 0.825 |
| otc_token_word | word | p_word_w0 | 0.038 | 0.012 | 3.3 | 0.530 |
| otc_token_word | word | neg_lam | -32.893 | -37.844 | - | 0.587 |
| otc_token_word | word | own_p | 0.023 | 0.009 | 2.6 | 0.526 |
| otc_token_word | bound | ins_w0 | 0.245 | 0.020 | 12.2 | 0.728 |
| otc_token_word_recip | char | p_tok_w0 | 0.310 | 0.028 | 11.2 | 0.821 |
| otc_token_word_recip | char | neg_lam | -1.518 | -7.210 | - | 0.830 |
| otc_token_word_recip | char | own_p | 0.279 | 0.022 | 12.7 | 0.811 |
| otc_token_word_recip | word | p_word_w0 | 0.040 | 0.009 | 4.5 | 0.536 |
| otc_token_word_recip | word | neg_lam | -35.775 | -40.826 | - | 0.588 |
| otc_token_word_recip | word | own_p | 0.013 | 0.005 | 2.7 | 0.514 |
| otc_token_word_recip | bound | ins_w0 | 0.312 | 0.025 | 12.5 | 0.745 |
| otc_word | char | p_tok_w0 | 0.221 | 0.022 | 10.1 | 0.722 |
| otc_word | char | neg_lam | -3.460 | -6.504 | - | 0.751 |
| otc_word | word | p_word_w0 | 0.022 | 0.006 | 3.7 | 0.527 |
| otc_word | word | neg_lam | -32.883 | -36.277 | - | 0.575 |
| otc_word | word | own_p | 0.026 | 0.006 | 4.5 | 0.528 |
| otc_word | bound | ins_w0 | 0.311 | 0.023 | 13.4 | 0.779 |
| otc_word_tuned | char | p_tok_w0 | 0.204 | 0.024 | 8.4 | 0.834 |
| otc_word_tuned | char | neg_lam | -2.586 | -5.458 | - | 0.856 |
| otc_word_tuned | word | p_word_w0 | 0.036 | 0.011 | 3.3 | 0.537 |
| otc_word_tuned | word | neg_lam | -29.166 | -34.553 | - | 0.597 |
| otc_word_tuned | word | own_p | 0.047 | 0.012 | 3.8 | 0.555 |
| otc_word_tuned | bound | ins_w0 | 0.219 | 0.015 | 15.0 | 0.729 |

**Share of units with star > 0.5** (disputed / undisputed; and how many of all starred units are disputed)

| model | char: disputed | char: undisputed | char: precision | word (p_word_w0): disputed | word: undisputed | boundary ins (w=0): disputed | boundary: undisputed |
|---|---|---|---|---|---|---|---|
| ctc | 0.162 (608) | 0.0117 (1179) | 0.340 | 0.024 | 0.0072 | 0.187 | 0.0029 |
| otc_recip | 0.287 (1074) | 0.0208 (2098) | 0.339 | 0.038 | 0.0078 | 0.282 | 0.0068 |
| otc_token | 0.244 (915) | 0.0175 (1768) | 0.341 | 0.035 | 0.0077 | 0.209 | 0.0044 |
| otc_token_word | 0.231 (866) | 0.0178 (1797) | 0.325 | 0.038 | 0.0113 | 0.198 | 0.0040 |
| otc_token_word_recip | 0.301 (1129) | 0.0224 (2260) | 0.333 | 0.040 | 0.0090 | 0.272 | 0.0066 |
| otc_word | 0.208 (780) | 0.0138 (1395) | 0.359 | 0.022 | 0.0058 | 0.277 | 0.0071 |
| otc_word_tuned | 0.187 (700) | 0.0144 (1451) | 0.325 | 0.035 | 0.0110 | 0.184 | 0.0033 |

**Disputed chars with star > 0.5, by character class** (token probe, w=0)

| class | n disputed | ctc | otc_recip | otc_token | otc_token_word | otc_token_word_recip | otc_word | otc_word_tuned |
|---|---|---|---|---|---|---|---|---|
| consonant | 1256 | 0.22 | 0.30 | 0.29 | 0.27 | 0.32 | 0.24 | 0.25 |
| indep_vowel | 1138 | 0.08 | 0.23 | 0.16 | 0.13 | 0.24 | 0.15 | 0.10 |
| vowel_sign | 912 | 0.19 | 0.32 | 0.27 | 0.28 | 0.35 | 0.24 | 0.21 |
| space | 198 | 0.11 | 0.25 | 0.23 | 0.20 | 0.22 | 0.15 | 0.14 |
| virama | 139 | 0.13 | 0.32 | 0.29 | 0.35 | 0.31 | 0.14 | 0.22 |
| other_mark | 105 | 0.27 | 0.46 | 0.37 | 0.36 | 0.43 | 0.34 | 0.25 |

**vs CTC control** (adjusted OR of star on disputed units; ratio = OTC/CTC; dAUC with 300-sample cluster bootstrap)

| model | unit | outcome | OR CTC | OR model | ratio (model/CTC) | dAUC(p) | dAUC(-lam) |
|---|---|---|---|---|---|---|---|
| otc_recip | char | p_tok_w0>0.5 | 11.45 [9.83, 13.33] p=1.1e-216 | 13.44 [11.99, 15.07] p=0.0e+00 | 1.17 [1.03, 1.33] p=1.4e-02 | -0.010 [-0.017, -0.004] | -0.017 [-0.024, -0.009] |
| otc_recip | word | p_word_w0>0.5 | 2.31 [1.55, 3.43] p=3.9e-05 | 4.54 [3.37, 6.12] p=3.6e-23 | 1.97 [1.47, 2.63] p=4.2e-06 | -0.003 [-0.007, +0.002] | -0.012 [-0.018, -0.007] |
| otc_recip | bound | ins_w0>0.5 | 66.37 [54.27, 81.16] p=0.0e+00 | 50.84 [43.23, 59.79] p=0.0e+00 | 0.77 [0.65, 0.90] p=1.3e-03 | +0.007 [-0.009, +0.022] | - |
| otc_token | char | p_tok_w0>0.5 | 11.64 [10.00, 13.55] p=2.6e-220 | 13.50 [11.89, 15.34] p=0.0e+00 | 1.16 [1.03, 1.30] p=1.2e-02 | +0.009 [+0.003, +0.015] | +0.001 [-0.004, +0.007] |
| otc_token | word | p_word_w0>0.5 | 2.46 [1.65, 3.68] p=1.1e-05 | 4.02 [2.98, 5.43] p=1.0e-19 | 1.63 [1.22, 2.18] p=9.0e-04 | -0.005 [-0.009, -0.002] | -0.010 [-0.015, -0.004] |
| otc_token | bound | ins_w0>0.5 | 69.59 [56.83, 85.22] p=0.0e+00 | 52.35 [43.29, 63.30] p=0.0e+00 | 0.75 [0.63, 0.89] p=1.1e-03 | +0.004 [-0.009, +0.017] | - |
| otc_token_word | char | p_tok_w0>0.5 | 11.85 [10.18, 13.79] p=5.1e-224 | 14.22 [12.50, 16.17] p=0.0e+00 | 1.20 [1.06, 1.35] p=3.3e-03 | +0.011 [+0.004, +0.018] | +0.004 [-0.001, +0.009] |
| otc_token_word | word | p_word_w0>0.5 | 2.67 [1.80, 3.95] p=9.1e-07 | 2.91 [2.25, 3.77] p=5.0e-16 | 1.09 [0.78, 1.52] p=6.1e-01 | -0.005 [-0.009, -0.001] | -0.003 [-0.008, +0.001] |
| otc_token_word | bound | ins_w0>0.5 | 66.94 [54.74, 81.87] p=0.0e+00 | 53.61 [43.83, 65.57] p=0.0e+00 | 0.80 [0.68, 0.95] p=9.2e-03 | -0.010 [-0.025, +0.005] | - |
| otc_token_word_recip | char | p_tok_w0>0.5 | 11.45 [9.84, 13.33] p=1.1e-216 | 13.86 [12.35, 15.55] p=0.0e+00 | 1.21 [1.06, 1.38] p=4.5e-03 | -0.011 [-0.018, -0.005] | -0.020 [-0.027, -0.013] |
| otc_token_word_recip | word | p_word_w0>0.5 | 2.63 [1.77, 3.91] p=1.6e-06 | 3.75 [2.89, 4.85] p=1.3e-23 | 1.42 [1.04, 1.94] p=2.5e-02 | +0.001 [-0.003, +0.006] | -0.002 [-0.008, +0.004] |
| otc_token_word_recip | bound | ins_w0>0.5 | 63.86 [52.43, 77.78] p=0.0e+00 | 48.76 [41.45, 57.35] p=0.0e+00 | 0.76 [0.64, 0.91] p=1.9e-03 | +0.006 [-0.007, +0.021] | - |
| otc_word | char | p_tok_w0>0.5 | 12.24 [10.53, 14.23] p=1.0e-232 | 13.94 [12.10, 16.05] p=5.5e-293 | 1.14 [1.02, 1.27] p=2.2e-02 | -0.110 [-0.119, -0.101] | -0.100 [-0.109, -0.091] |
| otc_word | word | p_word_w0>0.5 | 2.46 [1.66, 3.65] p=8.0e-06 | 3.59 [2.06, 6.26] p=6.2e-06 | 1.46 [1.01, 2.11] p=4.4e-02 | -0.007 [-0.012, -0.004] | -0.015 [-0.020, -0.011] |
| otc_word | bound | ins_w0>0.5 | 75.25 [61.56, 91.98] p=0.0e+00 | 44.94 [37.63, 53.66] p=0.0e+00 | 0.60 [0.49, 0.72] p=8.4e-08 | +0.041 [+0.026, +0.056] | - |
| otc_word_tuned | char | p_tok_w0>0.5 | 11.99 [10.31, 13.96] p=5.9e-226 | 13.13 [11.38, 15.16] p=5.5e-271 | 1.09 [0.97, 1.23] p=1.3e-01 | +0.002 [-0.003, +0.008] | +0.005 [-0.000, +0.011] |
| otc_word_tuned | word | p_word_w0>0.5 | 2.68 [1.81, 3.95] p=7.8e-07 | 2.80 [2.14, 3.66] p=5.6e-14 | 1.05 [0.76, 1.44] p=7.9e-01 | +0.002 [-0.002, +0.005] | +0.007 [+0.004, +0.011] |
| otc_word_tuned | bound | ins_w0>0.5 | 70.02 [57.23, 85.66] p=0.0e+00 | 59.82 [48.47, 73.81] p=0.0e+00 | 0.85 [0.72, 1.01] p=6.5e-02 | -0.009 [-0.022, +0.004] | - |

## mr (108556 chars, 387 disputed = 0.36%)

| model | unit | measure | mean(disputed) | mean(undisputed) | ratio | AUC |
|---|---|---|---|---|---|---|
| ctc | char | p_tok_w0 | 0.366 | 0.019 | 19.2 | 0.820 |
| ctc | char | neg_lam | -1.087 | -5.682 | - | 0.837 |
| ctc | word | p_word_w0 | 0.011 | 0.004 | 2.6 | 0.511 |
| ctc | word | neg_lam | -55.818 | -46.542 | - | 0.388 |
| ctc | bound | ins_w0 | 0.185 | 0.016 | 11.9 | 0.682 |
| otc_recip | char | p_tok_w0 | 0.420 | 0.021 | 20.0 | 0.808 |
| otc_recip | char | neg_lam | 0.684 | -7.059 | - | 0.828 |
| otc_recip | char | own_p | 0.399 | 0.019 | 21.4 | 0.805 |
| otc_recip | word | p_word_w0 | 0.029 | 0.006 | 5.2 | 0.522 |
| otc_recip | word | neg_lam | -59.706 | -49.907 | - | 0.396 |
| otc_recip | bound | ins_w0 | 0.220 | 0.023 | 9.8 | 0.736 |
| otc_token | char | p_tok_w0 | 0.414 | 0.020 | 20.5 | 0.818 |
| otc_token | char | neg_lam | 0.393 | -6.896 | - | 0.845 |
| otc_token | char | own_p | 0.406 | 0.019 | 21.6 | 0.816 |
| otc_token | word | p_word_w0 | 0.020 | 0.005 | 3.8 | 0.512 |
| otc_token | word | neg_lam | -59.002 | -49.485 | - | 0.394 |
| otc_token | bound | ins_w0 | 0.200 | 0.021 | 9.4 | 0.729 |
| otc_token_word | char | p_tok_w0 | 0.397 | 0.024 | 16.7 | 0.813 |
| otc_token_word | char | neg_lam | 0.068 | -6.525 | - | 0.837 |
| otc_token_word | char | own_p | 0.348 | 0.019 | 18.4 | 0.776 |
| otc_token_word | word | p_word_w0 | 0.051 | 0.014 | 3.6 | 0.530 |
| otc_token_word | word | neg_lam | -51.553 | -45.623 | - | 0.416 |
| otc_token_word | word | own_p | 0.035 | 0.011 | 3.3 | 0.520 |
| otc_token_word | bound | ins_w0 | 0.199 | 0.020 | 10.0 | 0.722 |
| otc_token_word_recip | char | p_tok_w0 | 0.416 | 0.021 | 20.1 | 0.810 |
| otc_token_word_recip | char | neg_lam | 0.381 | -6.967 | - | 0.835 |
| otc_token_word_recip | char | own_p | 0.379 | 0.017 | 21.9 | 0.803 |
| otc_token_word_recip | word | p_word_w0 | 0.024 | 0.006 | 4.3 | 0.520 |
| otc_token_word_recip | word | neg_lam | -57.818 | -48.467 | - | 0.397 |
| otc_token_word_recip | word | own_p | 0.005 | 0.003 | 2.0 | 0.508 |
| otc_token_word_recip | bound | ins_w0 | 0.206 | 0.022 | 9.4 | 0.741 |
| otc_word | char | p_tok_w0 | 0.338 | 0.019 | 17.4 | 0.805 |
| otc_word | char | neg_lam | -1.233 | -6.340 | - | 0.839 |
| otc_word | word | p_word_w0 | 0.015 | 0.004 | 3.7 | 0.514 |
| otc_word | word | neg_lam | -53.901 | -43.591 | - | 0.372 |
| otc_word | word | own_p | 0.027 | 0.004 | 6.5 | 0.522 |
| otc_word | bound | ins_w0 | 0.200 | 0.019 | 10.7 | 0.689 |
| otc_word_tuned | char | p_tok_w0 | 0.365 | 0.031 | 11.8 | 0.800 |
| otc_word_tuned | char | neg_lam | -1.049 | -5.553 | - | 0.828 |
| otc_word_tuned | word | p_word_w0 | 0.099 | 0.033 | 3.0 | 0.550 |
| otc_word_tuned | word | neg_lam | -42.197 | -42.142 | - | 0.447 |
| otc_word_tuned | word | own_p | 0.119 | 0.034 | 3.5 | 0.558 |
| otc_word_tuned | bound | ins_w0 | 0.151 | 0.015 | 10.0 | 0.624 |

**Share of units with star > 0.5** (disputed / undisputed; and how many of all starred units are disputed)

| model | char: disputed | char: undisputed | char: precision | word (p_word_w0): disputed | word: undisputed | boundary ins (w=0): disputed | boundary: undisputed |
|---|---|---|---|---|---|---|---|
| ctc | 0.351 (136) | 0.0100 (1085) | 0.111 | 0.010 | 0.0042 | 0.146 | 0.0029 |
| otc_recip | 0.419 (162) | 0.0155 (1677) | 0.088 | 0.028 | 0.0057 | 0.164 | 0.0050 |
| otc_token | 0.421 (163) | 0.0142 (1535) | 0.096 | 0.019 | 0.0052 | 0.144 | 0.0041 |
| otc_token_word | 0.385 (149) | 0.0167 (1809) | 0.076 | 0.050 | 0.0138 | 0.134 | 0.0037 |
| otc_token_word_recip | 0.411 (159) | 0.0153 (1653) | 0.088 | 0.024 | 0.0054 | 0.144 | 0.0043 |
| otc_word | 0.326 (126) | 0.0113 (1227) | 0.093 | 0.015 | 0.0039 | 0.134 | 0.0047 |
| otc_word_tuned | 0.362 (140) | 0.0210 (2271) | 0.058 | 0.100 | 0.0327 | 0.116 | 0.0028 |

**Disputed chars with star > 0.5, by character class** (token probe, w=0)

| class | n disputed | ctc | otc_recip | otc_token | otc_token_word | otc_token_word_recip | otc_word | otc_word_tuned |
|---|---|---|---|---|---|---|---|---|
| consonant | 169 | 0.27 | 0.34 | 0.34 | 0.33 | 0.34 | 0.24 | 0.33 |
| vowel_sign | 108 | 0.43 | 0.48 | 0.48 | 0.45 | 0.48 | 0.41 | 0.43 |
| virama | 44 | 0.45 | 0.52 | 0.52 | 0.45 | 0.48 | 0.36 | 0.32 |
| other_mark | 34 | 0.26 | 0.35 | 0.35 | 0.32 | 0.35 | 0.35 | 0.29 |
| indep_vowel | 21 | 0.57 | 0.67 | 0.67 | 0.52 | 0.67 | 0.52 | 0.57 |
| space | 11 | 0.36 | 0.36 | 0.36 | 0.18 | 0.27 | 0.18 | 0.27 |

**vs CTC control** (adjusted OR of star on disputed units; ratio = OTC/CTC; dAUC with 300-sample cluster bootstrap)

| model | unit | outcome | OR CTC | OR model | ratio (model/CTC) | dAUC(p) | dAUC(-lam) |
|---|---|---|---|---|---|---|---|
| otc_recip | char | p_tok_w0>0.5 | 28.71 [21.09, 39.08] p=5.8e-101 | 26.97 [20.53, 35.42] p=4.8e-124 | 0.94 [0.74, 1.19] p=6.0e-01 | -0.012 [-0.036, +0.013] | -0.009 [-0.029, +0.010] |
| otc_recip | word | p_word_w0>0.5 | 2.43 [1.05, 5.64] p=3.9e-02 | 5.96 [3.47, 10.21] p=8.7e-11 | 2.45 [1.22, 4.90] p=1.1e-02 | +0.012 [+0.006, +0.019] | +0.008 [-0.002, +0.019] |
| otc_recip | bound | ins_w0>0.5 | 18.98 [10.98, 32.82] p=5.6e-26 | 46.32 [35.02, 61.28] p=4.8e-159 | 2.44 [1.56, 3.83] p=1.0e-04 | +0.054 [+0.028, +0.086] | - |
| otc_token | char | p_tok_w0>0.5 | 29.82 [21.94, 40.54] p=2.9e-104 | 30.28 [22.69, 40.41] p=7.9e-119 | 1.02 [0.81, 1.27] p=8.9e-01 | -0.002 [-0.024, +0.019] | +0.008 [-0.010, +0.023] |
| otc_token | word | p_word_w0>0.5 | 2.44 [1.04, 5.72] p=4.0e-02 | 5.44 [2.94, 10.07] p=7.2e-08 | 2.23 [1.13, 4.37] p=2.0e-02 | +0.002 [-0.006, +0.009] | +0.007 [-0.002, +0.015] |
| otc_token | bound | ins_w0>0.5 | 17.17 [9.68, 30.47] p=2.6e-22 | 47.31 [35.36, 63.28] p=8.5e-149 | 2.75 [1.70, 4.46] p=3.6e-05 | +0.046 [+0.016, +0.078] | - |
| otc_token_word | char | p_tok_w0>0.5 | 28.06 [20.58, 38.25] p=9.1e-99 | 28.63 [21.41, 38.30] p=3.8e-113 | 1.02 [0.80, 1.30] p=8.7e-01 | -0.007 [-0.027, +0.013] | +0.000 [-0.017, +0.016] |
| otc_token_word | word | p_word_w0>0.5 | 1.60 [0.48, 5.32] p=4.4e-01 | 2.66 [1.76, 4.03] p=3.4e-06 | 1.67 [0.57, 4.90] p=3.5e-01 | +0.019 [+0.010, +0.028] | +0.028 [+0.014, +0.040] |
| otc_token_word | bound | ins_w0>0.5 | 18.28 [10.52, 31.76] p=6.4e-25 | 50.47 [37.96, 67.12] p=3.1e-160 | 2.76 [1.70, 4.48] p=3.8e-05 | +0.040 [+0.012, +0.070] | - |
| otc_token_word_recip | char | p_tok_w0>0.5 | 28.91 [21.23, 39.37] p=2.9e-101 | 29.76 [22.22, 39.86] p=9.7e-115 | 1.03 [0.82, 1.30] p=8.1e-01 | -0.010 [-0.032, +0.009] | -0.002 [-0.022, +0.016] |
| otc_token_word_recip | word | p_word_w0>0.5 | 2.35 [0.99, 5.58] p=5.4e-02 | 5.27 [2.99, 9.31] p=9.8e-09 | 2.25 [1.07, 4.72] p=3.3e-02 | +0.010 [+0.004, +0.016] | +0.009 [-0.000, +0.018] |
| otc_token_word_recip | bound | ins_w0>0.5 | 44.65 [31.75, 62.79] p=9.5e-106 | 25.88 [18.21, 36.79] p=1.5e-73 | 0.58 [0.44, 0.76] p=5.7e-05 | +0.059 [+0.034, +0.087] | - |
| otc_word | char | p_tok_w0>0.5 | 31.23 [23.07, 42.29] p=9.1e-110 | 23.45 [17.62, 31.20] p=6.0e-104 | 0.75 [0.60, 0.94] p=1.2e-02 | -0.015 [-0.036, +0.008] | +0.002 [-0.021, +0.023] |
| otc_word | word | p_word_w0>0.5 | 2.32 [0.96, 5.64] p=6.2e-02 | 4.62 [2.45, 8.70] p=2.2e-06 | 1.99 [0.98, 4.03] p=5.7e-02 | +0.004 [-0.002, +0.010] | -0.016 [-0.024, -0.007] |
| otc_word | bound | ins_w0>0.5 | 36.57 [25.22, 53.02] p=2.1e-80 | 24.11 [17.46, 33.28] p=2.1e-83 | 0.66 [0.48, 0.91] p=1.2e-02 | +0.006 [-0.021, +0.032] | - |
| otc_word_tuned | char | p_tok_w0>0.5 | 27.11 [19.82, 37.07] p=9.6e-95 | 27.24 [20.10, 36.90] p=6.4e-101 | 1.00 [0.77, 1.31] p=9.7e-01 | -0.019 [-0.040, +0.000] | -0.009 [-0.028, +0.011] |
| otc_word_tuned | word | p_word_w0>0.5 | 1.27 [0.29, 5.55] p=7.5e-01 | 1.85 [1.37, 2.51] p=6.5e-05 | 1.46 [0.36, 6.00] p=6.0e-01 | +0.039 [+0.026, +0.051] | +0.059 [+0.040, +0.078] |
| otc_word_tuned | bound | ins_w0>0.5 | 19.61 [11.64, 33.04] p=4.8e-29 | 55.87 [42.86, 72.83] p=2.2e-194 | 2.85 [1.74, 4.67] p=3.3e-05 | -0.059 [-0.089, -0.022] | - |

## mt (106945 chars, 4105 disputed = 3.84%)

| model | unit | measure | mean(disputed) | mean(undisputed) | ratio | AUC |
|---|---|---|---|---|---|---|
| ctc | char | p_tok_w0 | 0.176 | 0.019 | 9.3 | 0.803 |
| ctc | char | neg_lam | -3.109 | -5.569 | - | 0.808 |
| ctc | word | p_word_w0 | 0.012 | 0.004 | 2.9 | 0.522 |
| ctc | word | neg_lam | -32.880 | -38.693 | - | 0.608 |
| ctc | bound | ins_w0 | 0.128 | 0.014 | 9.2 | 0.681 |
| otc_recip | char | p_tok_w0 | 0.252 | 0.021 | 11.7 | 0.786 |
| otc_recip | char | neg_lam | -2.683 | -7.853 | - | 0.792 |
| otc_recip | char | own_p | 0.244 | 0.019 | 12.7 | 0.787 |
| otc_recip | word | p_word_w0 | 0.026 | 0.005 | 5.2 | 0.525 |
| otc_recip | word | neg_lam | -39.680 | -45.170 | - | 0.594 |
| otc_recip | bound | ins_w0 | 0.188 | 0.025 | 7.5 | 0.671 |
| otc_token | char | p_tok_w0 | 0.227 | 0.019 | 11.8 | 0.815 |
| otc_token | char | neg_lam | -2.789 | -7.461 | - | 0.818 |
| otc_token | char | own_p | 0.222 | 0.018 | 12.4 | 0.818 |
| otc_token | word | p_word_w0 | 0.018 | 0.004 | 3.9 | 0.520 |
| otc_token | word | neg_lam | -38.635 | -44.069 | - | 0.596 |
| otc_token | bound | ins_w0 | 0.156 | 0.022 | 7.0 | 0.669 |
| otc_token_word | char | p_tok_w0 | 0.223 | 0.019 | 11.6 | 0.821 |
| otc_token_word | char | neg_lam | -2.861 | -7.277 | - | 0.823 |
| otc_token_word | char | own_p | 0.209 | 0.017 | 12.6 | 0.821 |
| otc_token_word | word | p_word_w0 | 0.017 | 0.006 | 3.0 | 0.520 |
| otc_token_word | word | neg_lam | -37.275 | -42.656 | - | 0.598 |
| otc_token_word | word | own_p | 0.008 | 0.003 | 2.4 | 0.513 |
| otc_token_word | bound | ins_w0 | 0.158 | 0.022 | 7.3 | 0.669 |
| otc_token_word_recip | char | p_tok_w0 | 0.254 | 0.021 | 12.2 | 0.796 |
| otc_token_word_recip | char | neg_lam | -2.686 | -7.748 | - | 0.802 |
| otc_token_word_recip | char | own_p | 0.240 | 0.018 | 13.3 | 0.795 |
| otc_token_word_recip | word | p_word_w0 | 0.021 | 0.005 | 4.5 | 0.523 |
| otc_token_word_recip | word | neg_lam | -38.286 | -43.925 | - | 0.601 |
| otc_token_word_recip | word | own_p | 0.004 | 0.002 | 2.2 | 0.507 |
| otc_token_word_recip | bound | ins_w0 | 0.186 | 0.025 | 7.5 | 0.671 |
| otc_word | char | p_tok_w0 | 0.183 | 0.019 | 9.9 | 0.696 |
| otc_word | char | neg_lam | -4.022 | -6.813 | - | 0.723 |
| otc_word | word | p_word_w0 | 0.013 | 0.004 | 3.2 | 0.519 |
| otc_word | word | neg_lam | -33.941 | -38.299 | - | 0.599 |
| otc_word | word | own_p | 0.014 | 0.005 | 3.1 | 0.518 |
| otc_word | bound | ins_w0 | 0.194 | 0.023 | 8.4 | 0.763 |
| otc_word_tuned | char | p_tok_w0 | 0.179 | 0.019 | 9.3 | 0.794 |
| otc_word_tuned | char | neg_lam | -3.063 | -5.706 | - | 0.808 |
| otc_word_tuned | word | p_word_w0 | 0.016 | 0.005 | 3.3 | 0.521 |
| otc_word_tuned | word | neg_lam | -34.196 | -40.364 | - | 0.608 |
| otc_word_tuned | word | own_p | 0.022 | 0.006 | 3.6 | 0.535 |
| otc_word_tuned | bound | ins_w0 | 0.140 | 0.014 | 9.7 | 0.697 |

**Share of units with star > 0.5** (disputed / undisputed; and how many of all starred units are disputed)

| model | char: disputed | char: undisputed | char: precision | word (p_word_w0): disputed | word: undisputed | boundary ins (w=0): disputed | boundary: undisputed |
|---|---|---|---|---|---|---|---|
| ctc | 0.156 (641) | 0.0096 (984) | 0.394 | 0.012 | 0.0041 | 0.104 | 0.0025 |
| otc_recip | 0.247 (1013) | 0.0174 (1789) | 0.362 | 0.025 | 0.0047 | 0.160 | 0.0057 |
| otc_token | 0.214 (879) | 0.0143 (1474) | 0.374 | 0.018 | 0.0042 | 0.125 | 0.0040 |
| otc_token_word | 0.210 (864) | 0.0140 (1435) | 0.376 | 0.017 | 0.0060 | 0.127 | 0.0036 |
| otc_token_word_recip | 0.247 (1015) | 0.0167 (1717) | 0.372 | 0.021 | 0.0044 | 0.155 | 0.0053 |
| otc_word | 0.172 (705) | 0.0115 (1184) | 0.373 | 0.012 | 0.0038 | 0.160 | 0.0069 |
| otc_word_tuned | 0.160 (655) | 0.0097 (997) | 0.396 | 0.016 | 0.0047 | 0.113 | 0.0026 |

**Disputed chars with star > 0.5, by character class** (token probe, w=0)

| class | n disputed | ctc | otc_recip | otc_token | otc_token_word | otc_token_word_recip | otc_word | otc_word_tuned |
|---|---|---|---|---|---|---|---|---|
| vowel_sign | 1496 | 0.14 | 0.25 | 0.22 | 0.21 | 0.25 | 0.16 | 0.16 |
| consonant | 1123 | 0.20 | 0.28 | 0.25 | 0.25 | 0.27 | 0.21 | 0.20 |
| indep_vowel | 854 | 0.13 | 0.20 | 0.17 | 0.17 | 0.21 | 0.15 | 0.11 |
| other_mark | 471 | 0.12 | 0.24 | 0.18 | 0.19 | 0.22 | 0.14 | 0.12 |
| virama | 97 | 0.30 | 0.28 | 0.30 | 0.26 | 0.37 | 0.23 | 0.27 |
| space | 64 | 0.11 | 0.20 | 0.16 | 0.16 | 0.17 | 0.16 | 0.17 |

**vs CTC control** (adjusted OR of star on disputed units; ratio = OTC/CTC; dAUC with 300-sample cluster bootstrap)

| model | unit | outcome | OR CTC | OR model | ratio (model/CTC) | dAUC(p) | dAUC(-lam) |
|---|---|---|---|---|---|---|---|
| otc_recip | char | p_tok_w0>0.5 | 12.35 [10.60, 14.38] p=1.5e-229 | 11.78 [10.44, 13.29] p=0.0e+00 | 0.95 [0.85, 1.07] p=4.3e-01 | -0.018 [-0.023, -0.011] | -0.017 [-0.023, -0.011] |
| otc_recip | word | p_word_w0>0.5 | 1.88 [1.26, 2.79] p=2.0e-03 | 4.82 [3.57, 6.53] p=1.9e-24 | 2.57 [1.82, 3.64] p=1.0e-07 | +0.003 [-0.000, +0.006] | -0.015 [-0.020, -0.010] |
| otc_recip | bound | ins_w0>0.5 | 36.79 [29.06, 46.57] p=2.8e-197 | 27.38 [22.81, 32.86] p=2.2e-276 | 0.74 [0.60, 0.92] p=5.2e-03 | -0.009 [-0.022, +0.003] | - |
| otc_token | char | p_tok_w0>0.5 | 12.42 [10.67, 14.47] p=1.9e-229 | 11.97 [10.46, 13.68] p=8.2e-288 | 0.96 [0.86, 1.08] p=5.1e-01 | +0.012 [+0.006, +0.018] | +0.010 [+0.003, +0.015] |
| otc_token | word | p_word_w0>0.5 | 1.99 [1.34, 2.96] p=7.0e-04 | 3.78 [2.70, 5.29] p=1.0e-14 | 1.90 [1.35, 2.66] p=1.9e-04 | -0.002 [-0.004, +0.001] | -0.013 [-0.017, -0.008] |
| otc_token | bound | ins_w0>0.5 | 36.68 [29.15, 46.16] p=4.8e-207 | 31.15 [25.26, 38.41] p=4.0e-227 | 0.85 [0.70, 1.03] p=1.1e-01 | -0.011 [-0.023, +0.001] | - |
| otc_token_word | char | p_tok_w0>0.5 | 12.32 [10.56, 14.36] p=3.2e-225 | 12.97 [11.34, 14.83] p=3.9e-306 | 1.05 [0.94, 1.17] p=3.6e-01 | +0.018 [+0.011, +0.023] | +0.015 [+0.009, +0.020] |
| otc_token_word | word | p_word_w0>0.5 | 2.13 [1.43, 3.17] p=2.1e-04 | 2.55 [1.85, 3.51] p=8.4e-09 | 1.20 [0.87, 1.66] p=2.7e-01 | -0.002 [-0.005, +0.001] | -0.011 [-0.015, -0.006] |
| otc_token_word | bound | ins_w0>0.5 | 38.14 [30.12, 48.30] p=1.4e-200 | 37.07 [30.18, 45.54] p=1.3e-259 | 0.97 [0.81, 1.17] p=7.7e-01 | -0.011 [-0.023, +0.001] | - |
| otc_token_word_recip | char | p_tok_w0>0.5 | 12.32 [10.58, 14.35] p=1.4e-229 | 12.25 [10.83, 13.85] p=0.0e+00 | 0.99 [0.88, 1.12] p=9.2e-01 | -0.007 [-0.014, -0.001] | -0.007 [-0.012, -0.000] |
| otc_token_word_recip | word | p_word_w0>0.5 | 1.94 [1.30, 2.90] p=1.1e-03 | 4.21 [3.03, 5.86] p=1.2e-17 | 2.17 [1.52, 3.10] p=2.2e-05 | +0.001 [-0.002, +0.004] | -0.007 [-0.012, -0.002] |
| otc_token_word_recip | bound | ins_w0>0.5 | 36.83 [29.11, 46.60] p=2.0e-198 | 30.06 [24.99, 36.15] p=1.3e-285 | 0.82 [0.66, 1.01] p=5.8e-02 | -0.010 [-0.022, +0.003] | - |
| otc_word | char | p_tok_w0>0.5 | 12.76 [10.94, 14.88] p=2.5e-231 | 12.24 [10.62, 14.12] p=1.3e-259 | 0.96 [0.85, 1.08] p=4.9e-01 | -0.107 [-0.115, -0.097] | -0.085 [-0.094, -0.075] |
| otc_word | word | p_word_w0>0.5 | 2.03 [1.36, 3.01] p=4.8e-04 | 2.86 [1.98, 4.15] p=2.5e-08 | 1.41 [1.00, 1.99] p=4.7e-02 | -0.003 [-0.006, -0.000] | -0.010 [-0.014, -0.006] |
| otc_word | bound | ins_w0>0.5 | 38.84 [30.77, 49.04] p=6.2e-208 | 16.78 [14.04, 20.04] p=2.4e-212 | 0.43 [0.35, 0.53] p=5.2e-15 | +0.082 [+0.067, +0.095] | - |
| otc_word_tuned | char | p_tok_w0>0.5 | 12.95 [11.11, 15.10] p=3.5e-235 | 14.07 [12.07, 16.41] p=2.7e-249 | 1.09 [0.97, 1.21] p=1.4e-01 | -0.009 [-0.015, -0.003] | +0.000 [-0.005, +0.006] |
| otc_word_tuned | word | p_word_w0>0.5 | 1.93 [1.29, 2.89] p=1.4e-03 | 3.00 [2.14, 4.21] p=1.8e-10 | 1.56 [1.14, 2.13] p=5.8e-03 | -0.001 [-0.003, +0.002] | -0.000 [-0.003, +0.002] |
| otc_word_tuned | bound | ins_w0>0.5 | 38.51 [30.44, 48.71] p=1.5e-203 | 41.98 [33.19, 53.09] p=1.3e-213 | 1.09 [0.92, 1.29] p=3.1e-01 | +0.016 [+0.008, +0.025] | - |

## te (125783 chars, 1033 disputed = 0.82%)

| model | unit | measure | mean(disputed) | mean(undisputed) | ratio | AUC |
|---|---|---|---|---|---|---|
| ctc | char | p_tok_w0 | 0.286 | 0.021 | 13.5 | 0.833 |
| ctc | char | neg_lam | -1.621 | -5.200 | - | 0.859 |
| ctc | word | p_word_w0 | 0.023 | 0.006 | 4.1 | 0.512 |
| ctc | word | neg_lam | -54.986 | -49.102 | - | 0.426 |
| ctc | bound | ins_w0 | 0.195 | 0.013 | 15.2 | 0.729 |
| otc_recip | char | p_tok_w0 | 0.373 | 0.022 | 16.6 | 0.837 |
| otc_recip | char | neg_lam | -0.532 | -6.435 | - | 0.858 |
| otc_recip | char | own_p | 0.360 | 0.020 | 17.8 | 0.838 |
| otc_recip | word | p_word_w0 | 0.030 | 0.006 | 4.9 | 0.521 |
| otc_recip | word | neg_lam | -59.335 | -53.253 | - | 0.439 |
| otc_recip | bound | ins_w0 | 0.266 | 0.021 | 12.9 | 0.738 |
| otc_token | char | p_tok_w0 | 0.349 | 0.022 | 15.7 | 0.843 |
| otc_token | char | neg_lam | -0.573 | -6.186 | - | 0.861 |
| otc_token | char | own_p | 0.341 | 0.021 | 16.2 | 0.844 |
| otc_token | word | p_word_w0 | 0.026 | 0.006 | 4.3 | 0.514 |
| otc_token | word | neg_lam | -58.792 | -52.703 | - | 0.437 |
| otc_token | bound | ins_w0 | 0.226 | 0.019 | 12.0 | 0.746 |
| otc_token_word | char | p_tok_w0 | 0.328 | 0.024 | 13.4 | 0.849 |
| otc_token_word | char | neg_lam | -1.028 | -5.719 | - | 0.871 |
| otc_token_word | char | own_p | 0.304 | 0.021 | 14.5 | 0.829 |
| otc_token_word | word | p_word_w0 | 0.034 | 0.012 | 2.9 | 0.516 |
| otc_token_word | word | neg_lam | -52.643 | -48.038 | - | 0.441 |
| otc_token_word | word | own_p | 0.020 | 0.009 | 2.3 | 0.513 |
| otc_token_word | bound | ins_w0 | 0.217 | 0.017 | 12.9 | 0.739 |
| otc_token_word_recip | char | p_tok_w0 | 0.371 | 0.023 | 16.3 | 0.832 |
| otc_token_word_recip | char | neg_lam | -0.250 | -6.437 | - | 0.855 |
| otc_token_word_recip | char | own_p | 0.343 | 0.019 | 18.0 | 0.824 |
| otc_token_word_recip | word | p_word_w0 | 0.034 | 0.007 | 4.9 | 0.521 |
| otc_token_word_recip | word | neg_lam | -57.874 | -52.578 | - | 0.442 |
| otc_token_word_recip | word | own_p | 0.007 | 0.002 | 2.9 | 0.508 |
| otc_token_word_recip | bound | ins_w0 | 0.269 | 0.021 | 12.7 | 0.736 |
| otc_word | char | p_tok_w0 | 0.298 | 0.021 | 14.4 | 0.799 |
| otc_word | char | neg_lam | -1.819 | -5.879 | - | 0.824 |
| otc_word | word | p_word_w0 | 0.020 | 0.005 | 4.1 | 0.514 |
| otc_word | word | neg_lam | -55.889 | -49.167 | - | 0.422 |
| otc_word | word | own_p | 0.019 | 0.004 | 4.8 | 0.517 |
| otc_word | bound | ins_w0 | 0.244 | 0.018 | 13.6 | 0.772 |
| otc_word_tuned | char | p_tok_w0 | 0.282 | 0.025 | 11.5 | 0.844 |
| otc_word_tuned | char | neg_lam | -1.715 | -4.947 | - | 0.871 |
| otc_word_tuned | word | p_word_w0 | 0.031 | 0.013 | 2.4 | 0.519 |
| otc_word_tuned | word | neg_lam | -49.026 | -45.318 | - | 0.447 |
| otc_word_tuned | word | own_p | 0.038 | 0.014 | 2.6 | 0.527 |
| otc_word_tuned | bound | ins_w0 | 0.187 | 0.012 | 16.0 | 0.745 |

**Share of units with star > 0.5** (disputed / undisputed; and how many of all starred units are disputed)

| model | char: disputed | char: undisputed | char: precision | word (p_word_w0): disputed | word: undisputed | boundary ins (w=0): disputed | boundary: undisputed |
|---|---|---|---|---|---|---|---|
| ctc | 0.271 (280) | 0.0099 (1240) | 0.184 | 0.024 | 0.0055 | 0.154 | 0.0023 |
| otc_recip | 0.372 (384) | 0.0149 (1862) | 0.171 | 0.029 | 0.0059 | 0.212 | 0.0038 |
| otc_token | 0.335 (346) | 0.0141 (1765) | 0.164 | 0.027 | 0.0058 | 0.170 | 0.0032 |
| otc_token_word | 0.309 (319) | 0.0150 (1875) | 0.145 | 0.034 | 0.0116 | 0.160 | 0.0027 |
| otc_token_word_recip | 0.369 (381) | 0.0154 (1917) | 0.166 | 0.033 | 0.0067 | 0.224 | 0.0044 |
| otc_word | 0.283 (292) | 0.0109 (1357) | 0.177 | 0.019 | 0.0051 | 0.178 | 0.0042 |
| otc_word_tuned | 0.258 (267) | 0.0122 (1517) | 0.150 | 0.032 | 0.0130 | 0.141 | 0.0022 |

**Disputed chars with star > 0.5, by character class** (token probe, w=0)

| class | n disputed | ctc | otc_recip | otc_token | otc_token_word | otc_token_word_recip | otc_word | otc_word_tuned |
|---|---|---|---|---|---|---|---|---|
| vowel_sign | 436 | 0.24 | 0.35 | 0.32 | 0.28 | 0.34 | 0.26 | 0.25 |
| consonant | 375 | 0.35 | 0.41 | 0.38 | 0.37 | 0.42 | 0.33 | 0.31 |
| space | 119 | 0.21 | 0.38 | 0.28 | 0.27 | 0.32 | 0.24 | 0.17 |
| virama | 63 | 0.21 | 0.35 | 0.33 | 0.27 | 0.35 | 0.29 | 0.19 |
| indep_vowel | 29 | 0.21 | 0.31 | 0.31 | 0.24 | 0.34 | 0.21 | 0.24 |
| other_mark | 11 | 0.27 | 0.45 | 0.27 | 0.36 | 0.36 | 0.27 | 0.18 |

**vs CTC control** (adjusted OR of star on disputed units; ratio = OTC/CTC; dAUC with 300-sample cluster bootstrap)

| model | unit | outcome | OR CTC | OR model | ratio (model/CTC) | dAUC(p) | dAUC(-lam) |
|---|---|---|---|---|---|---|---|
| otc_recip | char | p_tok_w0>0.5 | 30.28 [24.88, 36.86] p=1.2e-253 | 33.72 [28.38, 40.08] p=0.0e+00 | 1.11 [0.96, 1.30] p=1.7e-01 | +0.004 [-0.007, +0.014] | -0.002 [-0.013, +0.009] |
| otc_recip | word | p_word_w0>0.5 | 5.69 [3.44, 9.41] p=1.2e-11 | 6.45 [4.14, 10.07] p=2.2e-16 | 1.13 [0.73, 1.76] p=5.8e-01 | +0.009 [+0.005, +0.013] | +0.013 [+0.006, +0.020] |
| otc_recip | bound | ins_w0>0.5 | 58.48 [45.10, 75.82] p=6.0e-207 | 54.31 [43.84, 67.28] p=1.5e-292 | 0.93 [0.76, 1.14] p=4.8e-01 | +0.009 [-0.007, +0.027] | - |
| otc_token | char | p_tok_w0>0.5 | 30.79 [25.32, 37.45] p=2.0e-258 | 29.29 [24.48, 35.03] p=1.4e-298 | 0.95 [0.83, 1.09] p=4.6e-01 | +0.010 [-0.000, +0.022] | +0.002 [-0.009, +0.013] |
| otc_token | word | p_word_w0>0.5 | 5.67 [3.44, 9.35] p=1.1e-11 | 7.00 [4.42, 11.08] p=9.8e-17 | 1.23 [0.83, 1.84] p=3.0e-01 | +0.002 [-0.002, +0.006] | +0.011 [+0.004, +0.019] |
| otc_token | bound | ins_w0>0.5 | 64.69 [50.20, 83.36] p=8.8e-228 | 54.46 [42.70, 69.45] p=1.1e-227 | 0.84 [0.70, 1.02] p=7.5e-02 | +0.017 [+0.001, +0.034] | - |
| otc_token_word | char | p_tok_w0>0.5 | 29.41 [24.10, 35.89] p=9.4e-243 | 25.86 [21.72, 30.79] p=3.2e-292 | 0.88 [0.76, 1.02] p=7.9e-02 | +0.015 [+0.005, +0.027] | +0.012 [+0.001, +0.021] |
| otc_token_word | word | p_word_w0>0.5 | 6.16 [3.76, 10.08] p=4.7e-13 | 3.02 [2.01, 4.53] p=1.0e-07 | 0.49 [0.33, 0.74] p=5.7e-04 | +0.005 [+0.001, +0.009] | +0.015 [+0.009, +0.022] |
| otc_token_word | bound | ins_w0>0.5 | 41.33 [31.20, 54.75] p=2.5e-148 | 80.43 [63.20, 102.35] p=1.2e-278 | 1.95 [1.56, 2.42] p=2.4e-09 | +0.009 [-0.007, +0.027] | - |
| otc_token_word_recip | char | p_tok_w0>0.5 | 30.57 [25.11, 37.22] p=3.3e-254 | 32.28 [27.08, 38.47] p=0.0e+00 | 1.06 [0.91, 1.22] p=4.7e-01 | -0.001 [-0.012, +0.011] | -0.004 [-0.015, +0.007] |
| otc_token_word_recip | word | p_word_w0>0.5 | 5.33 [3.18, 8.93] p=2.2e-10 | 6.95 [4.62, 10.46] p=1.3e-20 | 1.31 [0.87, 1.96] p=2.0e-01 | +0.009 [+0.005, +0.013] | +0.016 [+0.008, +0.023] |
| otc_token_word_recip | bound | ins_w0>0.5 | 60.69 [46.78, 78.75] p=1.0e-209 | 48.20 [38.91, 59.72] p=2.1e-275 | 0.79 [0.65, 0.97] p=2.6e-02 | +0.007 [-0.012, +0.025] | - |
| otc_word | char | p_tok_w0>0.5 | 30.75 [25.29, 37.39] p=3.7e-258 | 29.84 [24.80, 35.91] p=2.5e-283 | 0.97 [0.84, 1.12] p=6.8e-01 | -0.034 [-0.050, -0.018] | -0.035 [-0.049, -0.021] |
| otc_word | word | p_word_w0>0.5 | 5.93 [3.62, 9.72] p=1.5e-12 | 5.01 [3.01, 8.34] p=5.7e-10 | 0.84 [0.53, 1.34] p=4.7e-01 | +0.003 [-0.003, +0.007] | -0.004 [-0.011, +0.002] |
| otc_word | bound | ins_w0>0.5 | 58.06 [44.49, 75.75] p=1.2e-196 | 50.72 [39.65, 64.88] p=2.0e-214 | 0.87 [0.71, 1.07] p=1.9e-01 | +0.043 [+0.027, +0.060] | - |
| otc_word_tuned | char | p_tok_w0>0.5 | 26.15 [21.36, 32.01] p=8.1e-220 | 25.43 [21.13, 30.60] p=6.4e-257 | 0.97 [0.83, 1.13] p=7.2e-01 | +0.011 [-0.000, +0.021] | +0.012 [+0.002, +0.024] |
| otc_word_tuned | word | p_word_w0>0.5 | 6.30 [3.82, 10.40] p=5.9e-13 | 2.46 [1.67, 3.61] p=5.2e-06 | 0.39 [0.26, 0.58] p=3.6e-06 | +0.008 [+0.003, +0.012] | +0.021 [+0.015, +0.027] |
| otc_word_tuned | bound | ins_w0>0.5 | 57.87 [44.77, 74.81] p=8.6e-211 | 66.71 [50.73, 87.73] p=1.7e-198 | 1.15 [0.95, 1.40] p=1.5e-01 | +0.016 [+0.002, +0.030] | - |

## Pooled across languages (language fixed effects)

| model | unit | OR CTC | OR model | ratio (model/CTC) | langs |
|---|---|---|---|---|---|
| otc_recip | char | 17.25 [16.26, 18.30] p=0.0e+00 | 16.96 [16.12, 17.84] p=0.0e+00 | 0.98 [0.94, 1.03] p=4.5e-01 | bh bn ch hi kn mg mr mt te |
| otc_recip | word | 3.00 [2.55, 3.53] p=9.4e-40 | 5.39 [4.78, 6.07] p=1.7e-166 | 1.80 [1.57, 2.05] p=7.2e-18 | bh bn ch hi kn mg mr mt te |
| otc_token | char | 17.00 [16.03, 18.04] p=0.0e+00 | 17.64 [16.73, 18.61] p=0.0e+00 | 1.04 [0.99, 1.08] p=9.0e-02 | bh bn ch hi kn mg mr mt te |
| otc_token | word | 3.62 [3.11, 4.20] p=1.0e-63 | 5.14 [4.52, 5.84] p=1.2e-138 | 1.42 [1.26, 1.60] p=6.6e-09 | bh bn ch hi kn mg mr mt te |
| otc_token_word | char | 15.32 [14.44, 16.26] p=0.0e+00 | 16.83 [15.94, 17.77] p=0.0e+00 | 1.10 [1.05, 1.15] p=4.5e-05 | bh bn ch hi kn mg mr mt te |
| otc_token_word | word | 3.45 [2.99, 3.99] p=6.0e-64 | 3.06 [2.75, 3.40] p=4.5e-94 | 0.89 [0.78, 1.00] p=5.1e-02 | bh bn ch hi kn mg mr mt te |
| otc_token_word_recip | char | 17.23 [16.24, 18.28] p=0.0e+00 | 15.99 [15.19, 16.84] p=0.0e+00 | 0.93 [0.89, 0.97] p=1.1e-03 | bh bn ch hi kn mg mr mt te |
| otc_token_word_recip | word | 3.60 [3.10, 4.18] p=7.5e-63 | 4.64 [4.12, 5.23] p=1.2e-139 | 1.29 [1.14, 1.46] p=3.6e-05 | bh bn ch hi kn mg mr mt te |
| otc_word | char | 17.10 [16.12, 18.15] p=0.0e+00 | 16.22 [15.30, 17.19] p=0.0e+00 | 0.95 [0.91, 0.99] p=1.8e-02 | bh bn ch hi kn mg mr mt te |
| otc_word | word | 3.19 [2.73, 3.74] p=3.1e-47 | 3.88 [3.33, 4.52] p=2.5e-67 | 1.22 [1.06, 1.39] p=4.0e-03 | bh bn ch hi kn mg mr mt te |
| otc_word_tuned | char | 18.03 [17.00, 19.12] p=0.0e+00 | 15.54 [14.64, 16.49] p=0.0e+00 | 0.86 [0.82, 0.90] p=5.6e-11 | bh bn ch hi kn mg mr mt te |
| otc_word_tuned | word | 3.69 [3.20, 4.26] p=3.2e-71 | 2.64 [2.38, 2.94] p=7.3e-73 | 0.72 [0.64, 0.81] p=3.9e-08 | bh bn ch hi kn mg mr mt te |

