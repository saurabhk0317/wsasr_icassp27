# <star> placement vs validator disagreement (dev)

Excluded char classes: punct. Star measures: see analyze.py docstring.

## bh (72865 chars, 1046 disputed = 1.44%)

| model | unit | measure | mean(disputed) | mean(undisputed) | ratio | AUC |
|---|---|---|---|---|---|---|
| ctc | char | p_tok_w0 | 0.319 | 0.019 | 16.8 | 0.857 |
| ctc | char | neg_lam | -1.389 | -5.444 | - | 0.874 |
| ctc | word | p_word_w0 | 0.031 | 0.004 | 7.1 | 0.542 |
| ctc | word | neg_lam | -34.426 | -34.246 | - | 0.509 |
| ctc | bound | ins_w0 | 0.214 | 0.013 | 16.9 | 0.764 |
| otc_recip | char | p_tok_w0 | 0.395 | 0.021 | 18.4 | 0.850 |
| otc_recip | char | neg_lam | 0.297 | -7.414 | - | 0.874 |
| otc_recip | char | own_p | 0.380 | 0.019 | 19.9 | 0.854 |
| otc_recip | word | p_word_w0 | 0.045 | 0.007 | 6.7 | 0.542 |
| otc_recip | word | neg_lam | -40.537 | -39.751 | - | 0.502 |
| otc_recip | bound | ins_w0 | 0.333 | 0.023 | 14.2 | 0.768 |
| otc_token | char | p_tok_w0 | 0.384 | 0.019 | 20.0 | 0.871 |
| otc_token | char | neg_lam | 0.048 | -7.026 | - | 0.889 |
| otc_token | char | own_p | 0.378 | 0.018 | 20.5 | 0.873 |
| otc_token | word | p_word_w0 | 0.042 | 0.006 | 6.8 | 0.541 |
| otc_token | word | neg_lam | -40.022 | -39.115 | - | 0.500 |
| otc_token | bound | ins_w0 | 0.277 | 0.020 | 13.6 | 0.780 |
| otc_token_word | char | p_tok_w0 | 0.388 | 0.021 | 18.3 | 0.862 |
| otc_token_word | char | neg_lam | 0.250 | -6.861 | - | 0.884 |
| otc_token_word | char | own_p | 0.333 | 0.018 | 19.0 | 0.808 |
| otc_token_word | word | p_word_w0 | 0.046 | 0.008 | 6.0 | 0.545 |
| otc_token_word | word | neg_lam | -37.526 | -37.865 | - | 0.507 |
| otc_token_word | word | own_p | 0.027 | 0.005 | 5.0 | 0.528 |
| otc_token_word | bound | ins_w0 | 0.259 | 0.020 | 12.9 | 0.777 |
| otc_token_word_recip | char | p_tok_w0 | 0.392 | 0.020 | 19.3 | 0.857 |
| otc_token_word_recip | char | neg_lam | 0.321 | -7.257 | - | 0.875 |
| otc_token_word_recip | char | own_p | 0.337 | 0.016 | 20.5 | 0.803 |
| otc_token_word_recip | word | p_word_w0 | 0.048 | 0.006 | 7.9 | 0.542 |
| otc_token_word_recip | word | neg_lam | -39.434 | -39.102 | - | 0.502 |
| otc_token_word_recip | word | own_p | 0.021 | 0.003 | 7.4 | 0.514 |
| otc_token_word_recip | bound | ins_w0 | 0.304 | 0.022 | 13.7 | 0.772 |
| otc_word | char | p_tok_w0 | 0.325 | 0.018 | 17.6 | 0.834 |
| otc_word | char | neg_lam | -1.460 | -6.512 | - | 0.858 |
| otc_word | word | p_word_w0 | 0.029 | 0.005 | 6.1 | 0.535 |
| otc_word | word | neg_lam | -36.277 | -34.973 | - | 0.502 |
| otc_word | word | own_p | 0.040 | 0.006 | 6.5 | 0.545 |
| otc_word | bound | ins_w0 | 0.306 | 0.020 | 15.5 | 0.774 |
| otc_word_tuned | char | p_tok_w0 | 0.324 | 0.020 | 16.0 | 0.852 |
| otc_word_tuned | char | neg_lam | -1.137 | -5.508 | - | 0.873 |
| otc_word_tuned | word | p_word_w0 | 0.038 | 0.008 | 4.8 | 0.538 |
| otc_word_tuned | word | neg_lam | -35.298 | -36.261 | - | 0.519 |
| otc_word_tuned | word | own_p | 0.050 | 0.009 | 5.4 | 0.563 |
| otc_word_tuned | bound | ins_w0 | 0.245 | 0.013 | 18.3 | 0.752 |

**Share of units with star > 0.5** (disputed / undisputed; and how many of all starred units are disputed)

| model | char: disputed | char: undisputed | char: precision | word (p_word_w0): disputed | word: undisputed | boundary ins (w=0): disputed | boundary: undisputed |
|---|---|---|---|---|---|---|---|
| ctc | 0.305 (319) | 0.0091 (656) | 0.327 | 0.031 | 0.0041 | 0.171 | 0.0015 |
| otc_recip | 0.388 (406) | 0.0164 (1180) | 0.256 | 0.044 | 0.0066 | 0.276 | 0.0045 |
| otc_token | 0.370 (387) | 0.0131 (940) | 0.292 | 0.041 | 0.0061 | 0.230 | 0.0030 |
| otc_token_word | 0.380 (398) | 0.0150 (1080) | 0.269 | 0.045 | 0.0077 | 0.222 | 0.0026 |
| otc_token_word_recip | 0.382 (400) | 0.0150 (1076) | 0.271 | 0.048 | 0.0058 | 0.283 | 0.0039 |
| otc_word | 0.311 (325) | 0.0107 (770) | 0.297 | 0.028 | 0.0045 | 0.260 | 0.0044 |
| otc_word_tuned | 0.316 (331) | 0.0104 (745) | 0.308 | 0.037 | 0.0079 | 0.212 | 0.0018 |

**Disputed chars with star > 0.5, by character class** (token probe, w=0)

| class | n disputed | ctc | otc_recip | otc_token | otc_token_word | otc_token_word_recip | otc_word | otc_word_tuned |
|---|---|---|---|---|---|---|---|---|
| consonant | 366 | 0.29 | 0.39 | 0.34 | 0.35 | 0.34 | 0.30 | 0.33 |
| vowel_sign | 310 | 0.32 | 0.41 | 0.40 | 0.42 | 0.41 | 0.32 | 0.34 |
| indep_vowel | 183 | 0.26 | 0.32 | 0.30 | 0.33 | 0.35 | 0.23 | 0.21 |
| space | 99 | 0.22 | 0.32 | 0.35 | 0.29 | 0.35 | 0.28 | 0.20 |
| virama | 48 | 0.52 | 0.56 | 0.60 | 0.58 | 0.52 | 0.46 | 0.50 |
| other_mark | 40 | 0.47 | 0.53 | 0.53 | 0.55 | 0.57 | 0.53 | 0.57 |

**vs CTC control** (adjusted OR of star on disputed units; ratio = OTC/CTC; dAUC with 300-sample cluster bootstrap)

| model | unit | outcome | OR CTC | OR model | ratio (model/CTC) | dAUC(p) | dAUC(-lam) |
|---|---|---|---|---|---|---|---|
| otc_recip | char | p_tok_w0>0.5 | 27.03 [21.69, 33.67] p=1.0e-189 | 24.01 [19.72, 29.24] p=2.7e-219 | 0.89 [0.76, 1.04] p=1.5e-01 | -0.007 [-0.018, +0.004] | +0.000 [-0.011, +0.010] |
| otc_recip | word | p_word_w0>0.5 | 4.90 [2.88, 8.32] p=4.1e-09 | 6.12 [4.22, 8.88] p=1.4e-21 | 1.25 [0.82, 1.89] p=2.9e-01 | -0.000 [-0.007, +0.007] | -0.007 [-0.016, +0.005] |
| otc_recip | bound | ins_w0>0.5 | 96.51 [66.57, 139.91] p=1.8e-128 | 68.31 [51.84, 90.02] p=8.8e-198 | 0.71 [0.51, 0.98] p=3.9e-02 | +0.004 [-0.020, +0.032] | - |
| otc_token | char | p_tok_w0>0.5 | 24.75 [19.77, 31.00] p=5.6e-172 | 28.54 [23.26, 35.01] p=3.5e-226 | 1.15 [1.00, 1.33] p=4.9e-02 | +0.014 [+0.003, +0.025] | +0.016 [+0.005, +0.025] |
| otc_token | word | p_word_w0>0.5 | 4.99 [2.93, 8.49] p=3.1e-09 | 6.01 [4.03, 8.96] p=1.3e-18 | 1.21 [0.78, 1.85] p=3.9e-01 | -0.001 [-0.008, +0.006] | -0.009 [-0.017, -0.000] |
| otc_token | bound | ins_w0>0.5 | 62.68 [43.06, 91.25] p=2.0e-103 | 100.23 [74.77, 134.37] p=2.1e-208 | 1.60 [1.18, 2.17] p=2.7e-03 | +0.016 [-0.011, +0.043] | - |
| otc_token_word | char | p_tok_w0>0.5 | 27.55 [22.12, 34.31] p=1.0e-192 | 25.94 [21.48, 31.33] p=3.5e-250 | 0.94 [0.81, 1.09] p=4.3e-01 | +0.005 [-0.007, +0.016] | +0.010 [-0.000, +0.020] |
| otc_token_word | word | p_word_w0>0.5 | 5.48 [3.31, 9.09] p=4.0e-11 | 4.70 [3.16, 6.99] p=2.1e-14 | 0.86 [0.58, 1.26] p=4.3e-01 | +0.003 [-0.003, +0.010] | -0.002 [-0.010, +0.007] |
| otc_token_word | bound | ins_w0>0.5 | 75.67 [52.24, 109.60] p=6.3e-116 | 93.17 [67.59, 128.42] p=8.6e-169 | 1.23 [0.92, 1.65] p=1.7e-01 | +0.013 [-0.014, +0.034] | - |
| otc_token_word_recip | char | p_tok_w0>0.5 | 23.53 [18.75, 29.53] p=1.6e-163 | 27.15 [22.45, 32.85] p=2.8e-253 | 1.15 [0.98, 1.36] p=9.3e-02 | -0.001 [-0.013, +0.012] | +0.001 [-0.009, +0.011] |
| otc_token_word_recip | word | p_word_w0>0.5 | 4.80 [2.81, 8.17] p=8.2e-09 | 7.46 [5.12, 10.87] p=1.3e-25 | 1.56 [1.05, 2.30] p=2.6e-02 | +0.000 [-0.008, +0.008] | -0.007 [-0.016, +0.004] |
| otc_token_word_recip | bound | ins_w0>0.5 | 90.82 [62.83, 131.29] p=4.4e-127 | 81.63 [61.40, 108.54] p=1.6e-201 | 0.90 [0.67, 1.21] p=4.8e-01 | +0.008 [-0.016, +0.035] | - |
| otc_word | char | p_tok_w0>0.5 | 26.37 [21.12, 32.93] p=1.7e-183 | 26.08 [21.15, 32.16] p=3.3e-204 | 0.99 [0.84, 1.16] p=8.9e-01 | -0.024 [-0.036, -0.011] | -0.016 [-0.027, -0.004] |
| otc_word | word | p_word_w0>0.5 | 4.80 [2.85, 8.09] p=3.9e-09 | 4.23 [2.60, 6.90] p=7.0e-09 | 0.88 [0.62, 1.25] p=4.8e-01 | -0.007 [-0.015, -0.000] | -0.007 [-0.016, +0.002] |
| otc_word | bound | ins_w0>0.5 | 96.23 [65.62, 141.12] p=7.2e-121 | 61.17 [44.86, 83.42] p=5.7e-149 | 0.64 [0.45, 0.90] p=1.0e-02 | +0.010 [-0.018, +0.037] | - |
| otc_word_tuned | char | p_tok_w0>0.5 | 26.55 [21.26, 33.15] p=2.5e-184 | 29.28 [23.62, 36.29] p=8.5e-209 | 1.10 [0.94, 1.30] p=2.4e-01 | -0.006 [-0.018, +0.006] | -0.001 [-0.012, +0.008] |
| otc_word_tuned | word | p_word_w0>0.5 | 5.84 [3.59, 9.51] p=1.2e-12 | 3.31 [2.13, 5.14] p=1.0e-07 | 0.57 [0.39, 0.83] p=3.8e-03 | -0.003 [-0.010, +0.003] | +0.010 [+0.002, +0.016] |
| otc_word_tuned | bound | ins_w0>0.5 | 94.24 [66.21, 134.13] p=1.3e-140 | 135.71 [96.06, 191.73] p=1.1e-170 | 1.44 [1.09, 1.91] p=1.1e-02 | -0.012 [-0.039, +0.014] | - |

## bn (79959 chars, 1744 disputed = 2.18%)

| model | unit | measure | mean(disputed) | mean(undisputed) | ratio | AUC |
|---|---|---|---|---|---|---|
| ctc | char | p_tok_w0 | 0.232 | 0.022 | 10.5 | 0.749 |
| ctc | char | neg_lam | -2.336 | -5.382 | - | 0.793 |
| ctc | word | p_word_w0 | 0.033 | 0.007 | 4.6 | 0.523 |
| ctc | word | neg_lam | -42.747 | -39.018 | - | 0.460 |
| ctc | bound | ins_w0 | 0.175 | 0.014 | 12.2 | 0.693 |
| otc_recip | char | p_tok_w0 | 0.313 | 0.024 | 13.0 | 0.756 |
| otc_recip | char | neg_lam | -0.583 | -7.087 | - | 0.783 |
| otc_recip | char | own_p | 0.304 | 0.021 | 14.6 | 0.757 |
| otc_recip | word | p_word_w0 | 0.071 | 0.008 | 8.7 | 0.542 |
| otc_recip | word | neg_lam | -48.219 | -46.798 | - | 0.491 |
| otc_recip | bound | ins_w0 | 0.237 | 0.026 | 9.2 | 0.683 |
| otc_token | char | p_tok_w0 | 0.289 | 0.022 | 13.1 | 0.779 |
| otc_token | char | neg_lam | -1.336 | -6.713 | - | 0.802 |
| otc_token | char | own_p | 0.282 | 0.021 | 13.6 | 0.780 |
| otc_token | word | p_word_w0 | 0.047 | 0.008 | 6.0 | 0.530 |
| otc_token | word | neg_lam | -47.348 | -44.817 | - | 0.479 |
| otc_token | bound | ins_w0 | 0.202 | 0.022 | 9.1 | 0.683 |
| otc_token_word | char | p_tok_w0 | 0.308 | 0.029 | 10.6 | 0.767 |
| otc_token_word | char | neg_lam | -1.031 | -6.614 | - | 0.792 |
| otc_token_word | char | own_p | 0.255 | 0.020 | 12.7 | 0.723 |
| otc_token_word | word | p_word_w0 | 0.078 | 0.019 | 4.1 | 0.540 |
| otc_token_word | word | neg_lam | -41.080 | -41.819 | - | 0.494 |
| otc_token_word | word | own_p | 0.051 | 0.015 | 3.3 | 0.524 |
| otc_token_word | bound | ins_w0 | 0.210 | 0.022 | 9.4 | 0.657 |
| otc_token_word_recip | char | p_tok_w0 | 0.300 | 0.024 | 12.2 | 0.748 |
| otc_token_word_recip | char | neg_lam | -0.886 | -6.968 | - | 0.775 |
| otc_token_word_recip | char | own_p | 0.259 | 0.020 | 12.9 | 0.723 |
| otc_token_word_recip | word | p_word_w0 | 0.062 | 0.010 | 6.4 | 0.541 |
| otc_token_word_recip | word | neg_lam | -46.304 | -44.678 | - | 0.488 |
| otc_token_word_recip | word | own_p | 0.020 | 0.004 | 4.9 | 0.515 |
| otc_token_word_recip | bound | ins_w0 | 0.228 | 0.025 | 9.3 | 0.687 |
| otc_word | char | p_tok_w0 | 0.252 | 0.022 | 11.3 | 0.730 |
| otc_word | char | neg_lam | -2.378 | -6.264 | - | 0.774 |
| otc_word | word | p_word_w0 | 0.033 | 0.006 | 5.1 | 0.526 |
| otc_word | word | neg_lam | -44.791 | -40.822 | - | 0.454 |
| otc_word | word | own_p | 0.046 | 0.010 | 4.8 | 0.539 |
| otc_word | bound | ins_w0 | 0.223 | 0.021 | 10.5 | 0.735 |
| otc_word_tuned | char | p_tok_w0 | 0.257 | 0.027 | 9.4 | 0.742 |
| otc_word_tuned | char | neg_lam | -1.932 | -5.411 | - | 0.789 |
| otc_word_tuned | word | p_word_w0 | 0.066 | 0.015 | 4.4 | 0.537 |
| otc_word_tuned | word | neg_lam | -39.312 | -39.864 | - | 0.491 |
| otc_word_tuned | word | own_p | 0.085 | 0.017 | 5.2 | 0.553 |
| otc_word_tuned | bound | ins_w0 | 0.184 | 0.015 | 12.1 | 0.663 |

**Share of units with star > 0.5** (disputed / undisputed; and how many of all starred units are disputed)

| model | char: disputed | char: undisputed | char: precision | word (p_word_w0): disputed | word: undisputed | boundary ins (w=0): disputed | boundary: undisputed |
|---|---|---|---|---|---|---|---|
| ctc | 0.219 (382) | 0.0115 (900) | 0.298 | 0.032 | 0.0071 | 0.151 | 0.0024 |
| otc_recip | 0.308 (537) | 0.0184 (1440) | 0.272 | 0.071 | 0.0081 | 0.223 | 0.0062 |
| otc_token | 0.283 (494) | 0.0155 (1214) | 0.289 | 0.048 | 0.0077 | 0.181 | 0.0041 |
| otc_token_word | 0.300 (523) | 0.0224 (1749) | 0.230 | 0.078 | 0.0191 | 0.181 | 0.0043 |
| otc_token_word_recip | 0.292 (510) | 0.0186 (1452) | 0.260 | 0.060 | 0.0097 | 0.201 | 0.0053 |
| otc_word | 0.243 (424) | 0.0136 (1061) | 0.286 | 0.033 | 0.0063 | 0.196 | 0.0054 |
| otc_word_tuned | 0.245 (427) | 0.0170 (1330) | 0.243 | 0.066 | 0.0149 | 0.165 | 0.0029 |

**Disputed chars with star > 0.5, by character class** (token probe, w=0)

| class | n disputed | ctc | otc_recip | otc_token | otc_token_word | otc_token_word_recip | otc_word | otc_word_tuned |
|---|---|---|---|---|---|---|---|---|
| consonant | 726 | 0.26 | 0.34 | 0.33 | 0.34 | 0.33 | 0.29 | 0.29 |
| vowel_sign | 552 | 0.15 | 0.23 | 0.19 | 0.22 | 0.21 | 0.17 | 0.17 |
| virama | 237 | 0.16 | 0.28 | 0.23 | 0.25 | 0.24 | 0.15 | 0.17 |
| indep_vowel | 140 | 0.34 | 0.44 | 0.48 | 0.46 | 0.44 | 0.36 | 0.40 |
| space | 69 | 0.33 | 0.35 | 0.26 | 0.29 | 0.30 | 0.30 | 0.28 |
| other_mark | 20 | 0.35 | 0.70 | 0.55 | 0.65 | 0.70 | 0.50 | 0.30 |

**vs CTC control** (adjusted OR of star on disputed units; ratio = OTC/CTC; dAUC with 300-sample cluster bootstrap)

| model | unit | outcome | OR CTC | OR model | ratio (model/CTC) | dAUC(p) | dAUC(-lam) |
|---|---|---|---|---|---|---|---|
| otc_recip | char | p_tok_w0>0.5 | 17.09 [14.21, 20.57] p=2.1e-198 | 17.65 [15.00, 20.76] p=2.0e-262 | 1.03 [0.90, 1.18] p=6.4e-01 | +0.007 [-0.005, +0.019] | -0.010 [-0.021, +0.003] |
| otc_recip | word | p_word_w0>0.5 | 2.97 [1.86, 4.72] p=4.6e-06 | 8.13 [5.69, 11.63] p=1.4e-30 | 2.74 [1.85, 4.07] p=5.3e-07 | +0.020 [+0.013, +0.026] | +0.032 [+0.021, +0.042] |
| otc_recip | bound | ins_w0>0.5 | 81.01 [61.85, 106.10] p=1.2e-223 | 51.99 [42.35, 63.82] p=0.0e+00 | 0.64 [0.51, 0.81] p=1.3e-04 | -0.010 [-0.027, +0.006] | - |
| otc_token | char | p_tok_w0>0.5 | 17.48 [14.53, 21.03] p=4.1e-202 | 18.87 [15.88, 22.42] p=5.9e-245 | 1.08 [0.95, 1.23] p=2.5e-01 | +0.029 [+0.019, +0.039] | +0.009 [-0.001, +0.021] |
| otc_token | word | p_word_w0>0.5 | 2.99 [1.87, 4.78] p=4.7e-06 | 5.51 [3.68, 8.23] p=9.5e-17 | 1.84 [1.34, 2.52] p=1.5e-04 | +0.007 [+0.003, +0.012] | +0.019 [+0.012, +0.028] |
| otc_token | bound | ins_w0>0.5 | 83.50 [63.99, 108.96] p=6.4e-233 | 62.47 [49.25, 79.24] p=1.3e-254 | 0.75 [0.61, 0.92] p=7.0e-03 | -0.010 [-0.024, +0.010] | - |
| otc_token_word | char | p_tok_w0>0.5 | 16.74 [13.90, 20.16] p=3.1e-194 | 17.63 [14.91, 20.84] p=2.4e-247 | 1.05 [0.91, 1.21] p=4.7e-01 | +0.018 [+0.005, +0.028] | -0.001 [-0.012, +0.010] |
| otc_token_word | word | p_word_w0>0.5 | 3.60 [2.31, 5.62] p=1.6e-08 | 2.83 [2.10, 3.81] p=9.5e-12 | 0.78 [0.54, 1.14] p=2.0e-01 | +0.018 [+0.012, +0.024] | +0.034 [+0.024, +0.044] |
| otc_token_word | bound | ins_w0>0.5 | 74.60 [57.74, 96.38] p=1.2e-238 | 67.83 [53.70, 85.68] p=3.8e-274 | 0.91 [0.73, 1.14] p=4.1e-01 | -0.036 [-0.054, -0.018] | - |
| otc_token_word_recip | char | p_tok_w0>0.5 | 17.07 [14.19, 20.55] p=4.5e-198 | 17.03 [14.47, 20.06] p=1.1e-253 | 1.00 [0.87, 1.15] p=9.7e-01 | -0.001 [-0.015, +0.010] | -0.018 [-0.031, -0.007] |
| otc_token_word_recip | word | p_word_w0>0.5 | 3.58 [2.30, 5.57] p=1.5e-08 | 4.75 [3.26, 6.90] p=3.5e-16 | 1.33 [0.94, 1.87] p=1.1e-01 | +0.018 [+0.012, +0.023] | +0.029 [+0.020, +0.037] |
| otc_token_word_recip | bound | ins_w0>0.5 | 80.24 [61.41, 104.84] p=1.4e-226 | 54.92 [44.01, 68.54] p=3.3e-275 | 0.68 [0.54, 0.87] p=1.7e-03 | -0.006 [-0.024, +0.012] | - |
| otc_word | char | p_tok_w0>0.5 | 17.69 [14.70, 21.29] p=4.4e-203 | 17.32 [14.54, 20.65] p=8.2e-223 | 0.98 [0.85, 1.12] p=7.7e-01 | -0.019 [-0.032, -0.008] | -0.019 [-0.032, -0.006] |
| otc_word | word | p_word_w0>0.5 | 3.06 [1.92, 4.89] p=2.8e-06 | 4.28 [2.76, 6.65] p=9.3e-11 | 1.40 [0.99, 1.98] p=5.8e-02 | +0.003 [-0.001, +0.008] | -0.005 [-0.012, +0.002] |
| otc_word | bound | ins_w0>0.5 | 75.98 [58.18, 99.23] p=8.7e-222 | 49.40 [39.30, 62.10] p=9.9e-245 | 0.65 [0.51, 0.83] p=6.7e-04 | +0.042 [+0.022, +0.059] | - |
| otc_word_tuned | char | p_tok_w0>0.5 | 17.90 [14.87, 21.56] p=2.4e-203 | 15.65 [13.17, 18.58] p=8.3e-216 | 0.87 [0.76, 1.00] p=4.9e-02 | -0.007 [-0.017, +0.004] | -0.004 [-0.014, +0.007] |
| otc_word_tuned | word | p_word_w0>0.5 | 3.57 [2.30, 5.56] p=1.6e-08 | 2.40 [1.75, 3.28] p=5.1e-08 | 0.67 [0.47, 0.96] p=2.9e-02 | +0.014 [+0.008, +0.020] | +0.031 [+0.023, +0.041] |
| otc_word_tuned | bound | ins_w0>0.5 | 83.17 [63.63, 108.71] p=1.3e-229 | 78.80 [60.64, 102.41] p=5.4e-234 | 0.95 [0.75, 1.19] p=6.4e-01 | -0.030 [-0.048, -0.013] | - |

## ch (74184 chars, 344 disputed = 0.46%)

| model | unit | measure | mean(disputed) | mean(undisputed) | ratio | AUC |
|---|---|---|---|---|---|---|
| ctc | char | p_tok_w0 | 0.165 | 0.016 | 10.5 | 0.759 |
| ctc | char | neg_lam | -3.279 | -5.847 | - | 0.781 |
| ctc | word | p_word_w0 | 0.011 | 0.002 | 5.2 | 0.507 |
| ctc | word | neg_lam | -44.872 | -37.835 | - | 0.409 |
| ctc | bound | ins_w0 | 0.167 | 0.015 | 11.4 | 0.685 |
| otc_recip | char | p_tok_w0 | 0.209 | 0.015 | 14.2 | 0.766 |
| otc_recip | char | neg_lam | -3.802 | -8.259 | - | 0.761 |
| otc_recip | char | own_p | 0.199 | 0.014 | 14.3 | 0.766 |
| otc_recip | word | p_word_w0 | 0.027 | 0.002 | 12.4 | 0.516 |
| otc_recip | word | neg_lam | -52.134 | -42.664 | - | 0.401 |
| otc_recip | bound | ins_w0 | 0.221 | 0.025 | 9.0 | 0.731 |
| otc_token | char | p_tok_w0 | 0.222 | 0.014 | 15.5 | 0.775 |
| otc_token | char | neg_lam | -3.527 | -8.194 | - | 0.772 |
| otc_token | char | own_p | 0.213 | 0.014 | 15.7 | 0.773 |
| otc_token | word | p_word_w0 | 0.022 | 0.002 | 11.2 | 0.517 |
| otc_token | word | neg_lam | -51.594 | -42.582 | - | 0.404 |
| otc_token | bound | ins_w0 | 0.202 | 0.024 | 8.4 | 0.707 |
| otc_token_word | char | p_tok_w0 | 0.220 | 0.014 | 15.3 | 0.798 |
| otc_token_word | char | neg_lam | -3.356 | -7.899 | - | 0.794 |
| otc_token_word | char | own_p | 0.198 | 0.013 | 14.9 | 0.792 |
| otc_token_word | word | p_word_w0 | 0.016 | 0.002 | 6.4 | 0.515 |
| otc_token_word | word | neg_lam | -49.765 | -41.020 | - | 0.403 |
| otc_token_word | word | own_p | 0.007 | 0.001 | 5.4 | 0.508 |
| otc_token_word | bound | ins_w0 | 0.231 | 0.024 | 9.8 | 0.759 |
| otc_token_word_recip | char | p_tok_w0 | 0.219 | 0.015 | 14.4 | 0.758 |
| otc_token_word_recip | char | neg_lam | -3.526 | -8.057 | - | 0.758 |
| otc_token_word_recip | char | own_p | 0.203 | 0.014 | 14.6 | 0.752 |
| otc_token_word_recip | word | p_word_w0 | 0.024 | 0.002 | 10.8 | 0.521 |
| otc_token_word_recip | word | neg_lam | -49.555 | -40.907 | - | 0.399 |
| otc_token_word_recip | word | own_p | 0.007 | 0.001 | 6.0 | 0.510 |
| otc_token_word_recip | bound | ins_w0 | 0.239 | 0.024 | 9.9 | 0.751 |
| otc_word | char | p_tok_w0 | 0.167 | 0.015 | 11.5 | 0.700 |
| otc_word | char | neg_lam | -4.265 | -6.967 | - | 0.732 |
| otc_word | word | p_word_w0 | 0.009 | 0.001 | 6.3 | 0.510 |
| otc_word | word | neg_lam | -42.555 | -34.519 | - | 0.386 |
| otc_word | word | own_p | 0.009 | 0.002 | 5.1 | 0.515 |
| otc_word | bound | ins_w0 | 0.206 | 0.021 | 9.9 | 0.753 |
| otc_word_tuned | char | p_tok_w0 | 0.177 | 0.015 | 11.8 | 0.766 |
| otc_word_tuned | char | neg_lam | -3.251 | -6.120 | - | 0.786 |
| otc_word_tuned | word | p_word_w0 | 0.010 | 0.002 | 4.7 | 0.512 |
| otc_word_tuned | word | neg_lam | -45.109 | -38.800 | - | 0.421 |
| otc_word_tuned | word | own_p | 0.020 | 0.003 | 7.7 | 0.526 |
| otc_word_tuned | bound | ins_w0 | 0.224 | 0.016 | 13.8 | 0.731 |

**Share of units with star > 0.5** (disputed / undisputed; and how many of all starred units are disputed)

| model | char: disputed | char: undisputed | char: precision | word (p_word_w0): disputed | word: undisputed | boundary ins (w=0): disputed | boundary: undisputed |
|---|---|---|---|---|---|---|---|
| ctc | 0.142 (49) | 0.0070 (518) | 0.086 | 0.011 | 0.0019 | 0.128 | 0.0015 |
| otc_recip | 0.201 (69) | 0.0109 (807) | 0.079 | 0.028 | 0.0021 | 0.165 | 0.0030 |
| otc_token | 0.212 (73) | 0.0101 (749) | 0.089 | 0.023 | 0.0017 | 0.150 | 0.0027 |
| otc_token_word | 0.212 (73) | 0.0100 (735) | 0.090 | 0.016 | 0.0024 | 0.173 | 0.0023 |
| otc_token_word_recip | 0.212 (73) | 0.0112 (826) | 0.081 | 0.023 | 0.0021 | 0.173 | 0.0027 |
| otc_word | 0.154 (53) | 0.0076 (564) | 0.086 | 0.009 | 0.0013 | 0.173 | 0.0039 |
| otc_word_tuned | 0.154 (53) | 0.0071 (523) | 0.092 | 0.009 | 0.0019 | 0.173 | 0.0017 |

**Disputed chars with star > 0.5, by character class** (token probe, w=0)

| class | n disputed | ctc | otc_recip | otc_token | otc_token_word | otc_token_word_recip | otc_word | otc_word_tuned |
|---|---|---|---|---|---|---|---|---|
| consonant | 169 | 0.16 | 0.20 | 0.22 | 0.21 | 0.22 | 0.18 | 0.19 |
| vowel_sign | 101 | 0.11 | 0.19 | 0.18 | 0.18 | 0.18 | 0.09 | 0.11 |
| indep_vowel | 35 | 0.29 | 0.37 | 0.37 | 0.37 | 0.37 | 0.31 | 0.26 |
| other_mark | 30 | 0.03 | 0.10 | 0.13 | 0.13 | 0.07 | 0.03 | 0.00 |
| space | 6 | 0.00 | 0.17 | 0.17 | 0.33 | 0.33 | 0.33 | 0.17 |
| virama | 3 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

**vs CTC control** (adjusted OR of star on disputed units; ratio = OTC/CTC; dAUC with 300-sample cluster bootstrap)

| model | unit | outcome | OR CTC | OR model | ratio (model/CTC) | dAUC(p) | dAUC(-lam) |
|---|---|---|---|---|---|---|---|
| otc_recip | char | p_tok_w0>0.5 | 7.81 [4.17, 14.63] p=1.4e-10 | 18.84 [13.02, 27.27] p=1.2e-54 | 2.41 [1.49, 3.91] p=3.6e-04 | +0.007 [-0.020, +0.031] | -0.019 [-0.041, +0.003] |
| otc_recip | word | p_word_w0>0.5 | 4.86 [1.49, 15.85] p=8.8e-03 | 17.13 [8.55, 34.35] p=1.2e-15 | 3.52 [1.19, 10.40] p=2.2e-02 | +0.009 [+0.000, +0.015] | -0.008 [-0.022, +0.005] |
| otc_recip | bound | ins_w0>0.5 | 21.01 [5.26, 83.95] p=1.7e-05 | 93.20 [59.76, 145.37] p=6.1e-89 | 4.44 [1.33, 14.80] p=1.5e-02 | +0.045 [-0.005, +0.104] | - |
| otc_token | char | p_tok_w0>0.5 | 8.44 [4.64, 15.37] p=2.9e-12 | 22.75 [16.21, 31.92] p=5.4e-73 | 2.69 [1.68, 4.32] p=3.9e-05 | +0.016 [-0.009, +0.039] | -0.008 [-0.030, +0.012] |
| otc_token | word | p_word_w0>0.5 | 5.02 [1.66, 15.21] p=4.4e-03 | 16.56 [7.96, 34.48] p=6.2e-14 | 3.30 [1.16, 9.40] p=2.5e-02 | +0.010 [+0.004, +0.016] | -0.005 [-0.020, +0.009] |
| otc_token | bound | ins_w0>0.5 | 19.62 [4.66, 82.54] p=4.9e-05 | 74.27 [44.86, 122.96] p=5.9e-63 | 3.79 [1.15, 12.43] p=2.8e-02 | +0.022 [-0.021, +0.066] | - |
| otc_token_word | char | p_tok_w0>0.5 | 8.37 [4.59, 15.27] p=4.2e-12 | 21.74 [14.97, 31.56] p=6.9e-59 | 2.60 [1.69, 3.99] p=1.3e-05 | +0.039 [+0.016, +0.062] | +0.013 [-0.009, +0.035] |
| otc_token_word | word | p_word_w0>0.5 | 3.44 [0.74, 15.94] p=1.1e-01 | 7.09 [3.29, 15.28] p=5.6e-07 | 2.06 [0.63, 6.77] p=2.3e-01 | +0.008 [+0.000, +0.016] | -0.006 [-0.017, +0.004] |
| otc_token_word | bound | ins_w0>0.5 | 24.23 [7.45, 78.81] p=1.2e-07 | 94.85 [56.62, 158.89] p=5.1e-67 | 3.91 [1.62, 9.48] p=2.5e-03 | +0.073 [+0.030, +0.121] | - |
| otc_token_word_recip | char | p_tok_w0>0.5 | 8.01 [4.35, 14.74] p=2.3e-11 | 18.67 [13.06, 26.69] p=5.6e-58 | 2.33 [1.52, 3.58] p=1.1e-04 | -0.001 [-0.032, +0.030] | -0.023 [-0.046, +0.003] |
| otc_token_word_recip | word | p_word_w0>0.5 | 3.90 [1.00, 15.27] p=5.1e-02 | 10.07 [4.57, 22.17] p=9.7e-09 | 2.58 [0.82, 8.15] p=1.1e-01 | +0.014 [+0.005, +0.023] | -0.010 [-0.023, +0.002] |
| otc_token_word_recip | bound | ins_w0>0.5 | 21.33 [5.40, 84.34] p=1.3e-05 | 94.26 [60.84, 146.05] p=4.7e-92 | 4.42 [1.39, 14.10] p=1.2e-02 | +0.065 [+0.022, +0.114] | - |
| otc_word | char | p_tok_w0>0.5 | 9.25 [5.27, 16.23] p=9.0e-15 | 20.85 [14.69, 29.59] p=6.9e-65 | 2.25 [1.54, 3.30] p=3.1e-05 | -0.059 [-0.090, -0.027] | -0.048 [-0.078, -0.018] |
| otc_word | word | p_word_w0>0.5 | 4.76 [1.35, 16.79] p=1.5e-02 | 9.36 [3.50, 25.07] p=8.5e-06 | 1.97 [0.70, 5.54] p=2.0e-01 | +0.003 [-0.003, +0.008] | -0.023 [-0.036, -0.011] |
| otc_word | bound | ins_w0>0.5 | 16.26 [3.30, 80.12] p=6.1e-04 | 55.76 [34.96, 88.92] p=5.6e-64 | 3.43 [0.90, 13.14] p=7.2e-02 | +0.068 [+0.025, +0.108] | - |
| otc_word_tuned | char | p_tok_w0>0.5 | 9.76 [5.64, 16.88] p=3.8e-16 | 21.15 [14.57, 30.71] p=6.4e-58 | 2.17 [1.56, 3.02] p=4.7e-06 | +0.007 [-0.016, +0.028] | +0.005 [-0.020, +0.025] |
| otc_word_tuned | word | p_word_w0>0.5 | 3.69 [0.81, 16.72] p=9.0e-02 | 6.58 [2.54, 17.01] p=1.0e-04 | 1.78 [0.56, 5.65] p=3.3e-01 | +0.005 [+0.000, +0.011] | +0.012 [+0.004, +0.020] |
| otc_word_tuned | bound | ins_w0>0.5 | 24.33 [7.23, 81.88] p=2.5e-07 | 117.19 [72.66, 189.01] p=5.9e-85 | 4.82 [1.90, 12.22] p=9.3e-04 | +0.046 [+0.005, +0.090] | - |

## hi (83053 chars, 466 disputed = 0.56%)

| model | unit | measure | mean(disputed) | mean(undisputed) | ratio | AUC |
|---|---|---|---|---|---|---|
| ctc | char | p_tok_w0 | 0.337 | 0.022 | 15.5 | 0.788 |
| ctc | char | neg_lam | -1.523 | -5.533 | - | 0.813 |
| ctc | word | p_word_w0 | 0.031 | 0.005 | 5.9 | 0.531 |
| ctc | word | neg_lam | -43.284 | -36.181 | - | 0.422 |
| ctc | bound | ins_w0 | 0.326 | 0.014 | 23.6 | 0.826 |
| otc_recip | char | p_tok_w0 | 0.376 | 0.023 | 16.4 | 0.767 |
| otc_recip | char | neg_lam | -0.138 | -7.333 | - | 0.787 |
| otc_recip | char | own_p | 0.355 | 0.020 | 17.5 | 0.764 |
| otc_recip | word | p_word_w0 | 0.065 | 0.007 | 9.6 | 0.540 |
| otc_recip | word | neg_lam | -52.523 | -41.703 | - | 0.394 |
| otc_recip | bound | ins_w0 | 0.344 | 0.023 | 14.9 | 0.815 |
| otc_token | char | p_tok_w0 | 0.398 | 0.022 | 18.0 | 0.770 |
| otc_token | char | neg_lam | 0.233 | -7.270 | - | 0.781 |
| otc_token | char | own_p | 0.378 | 0.021 | 18.3 | 0.764 |
| otc_token | word | p_word_w0 | 0.062 | 0.006 | 10.2 | 0.542 |
| otc_token | word | neg_lam | -52.085 | -41.559 | - | 0.403 |
| otc_token | bound | ins_w0 | 0.386 | 0.022 | 17.5 | 0.801 |
| otc_token_word | char | p_tok_w0 | 0.383 | 0.026 | 14.8 | 0.781 |
| otc_token_word | char | neg_lam | -0.586 | -6.887 | - | 0.800 |
| otc_token_word | char | own_p | 0.302 | 0.019 | 16.0 | 0.727 |
| otc_token_word | word | p_word_w0 | 0.085 | 0.014 | 6.2 | 0.551 |
| otc_token_word | word | neg_lam | -40.819 | -36.633 | - | 0.450 |
| otc_token_word | word | own_p | 0.049 | 0.011 | 4.5 | 0.530 |
| otc_token_word | bound | ins_w0 | 0.298 | 0.021 | 14.3 | 0.787 |
| otc_token_word_recip | char | p_tok_w0 | 0.386 | 0.023 | 16.9 | 0.757 |
| otc_token_word_recip | char | neg_lam | -0.235 | -7.144 | - | 0.775 |
| otc_token_word_recip | char | own_p | 0.335 | 0.018 | 18.1 | 0.724 |
| otc_token_word_recip | word | p_word_w0 | 0.074 | 0.008 | 9.4 | 0.548 |
| otc_token_word_recip | word | neg_lam | -45.807 | -38.361 | - | 0.428 |
| otc_token_word_recip | word | own_p | 0.017 | 0.004 | 4.8 | 0.519 |
| otc_token_word_recip | bound | ins_w0 | 0.343 | 0.023 | 15.1 | 0.792 |
| otc_word | char | p_tok_w0 | 0.339 | 0.022 | 15.5 | 0.774 |
| otc_word | char | neg_lam | -1.403 | -6.431 | - | 0.799 |
| otc_word | word | p_word_w0 | 0.044 | 0.006 | 7.3 | 0.541 |
| otc_word | word | neg_lam | -41.880 | -33.911 | - | 0.423 |
| otc_word | word | own_p | 0.056 | 0.009 | 6.6 | 0.555 |
| otc_word | bound | ins_w0 | 0.441 | 0.019 | 23.5 | 0.850 |
| otc_word_tuned | char | p_tok_w0 | 0.382 | 0.030 | 12.6 | 0.783 |
| otc_word_tuned | char | neg_lam | -0.886 | -5.967 | - | 0.798 |
| otc_word_tuned | word | p_word_w0 | 0.099 | 0.022 | 4.5 | 0.562 |
| otc_word_tuned | word | neg_lam | -36.931 | -35.077 | - | 0.471 |
| otc_word_tuned | word | own_p | 0.116 | 0.025 | 4.7 | 0.577 |
| otc_word_tuned | bound | ins_w0 | 0.312 | 0.016 | 19.4 | 0.765 |

**Share of units with star > 0.5** (disputed / undisputed; and how many of all starred units are disputed)

| model | char: disputed | char: undisputed | char: precision | word (p_word_w0): disputed | word: undisputed | boundary ins (w=0): disputed | boundary: undisputed |
|---|---|---|---|---|---|---|---|
| ctc | 0.333 (155) | 0.0127 (1050) | 0.129 | 0.029 | 0.0051 | 0.245 | 0.0026 |
| otc_recip | 0.378 (176) | 0.0180 (1487) | 0.106 | 0.066 | 0.0067 | 0.260 | 0.0053 |
| otc_token | 0.397 (185) | 0.0168 (1390) | 0.117 | 0.066 | 0.0060 | 0.286 | 0.0047 |
| otc_token_word | 0.386 (180) | 0.0200 (1652) | 0.098 | 0.083 | 0.0138 | 0.250 | 0.0043 |
| otc_token_word_recip | 0.380 (177) | 0.0177 (1463) | 0.108 | 0.073 | 0.0079 | 0.296 | 0.0052 |
| otc_word | 0.337 (157) | 0.0143 (1180) | 0.117 | 0.042 | 0.0057 | 0.347 | 0.0047 |
| otc_word_tuned | 0.378 (176) | 0.0223 (1845) | 0.087 | 0.100 | 0.0220 | 0.250 | 0.0030 |

**Disputed chars with star > 0.5, by character class** (token probe, w=0)

| class | n disputed | ctc | otc_recip | otc_token | otc_token_word | otc_token_word_recip | otc_word | otc_word_tuned |
|---|---|---|---|---|---|---|---|---|
| consonant | 199 | 0.35 | 0.37 | 0.39 | 0.40 | 0.37 | 0.32 | 0.39 |
| vowel_sign | 123 | 0.36 | 0.39 | 0.39 | 0.40 | 0.41 | 0.38 | 0.36 |
| other_mark | 49 | 0.37 | 0.53 | 0.63 | 0.51 | 0.59 | 0.47 | 0.57 |
| virama | 48 | 0.23 | 0.38 | 0.35 | 0.35 | 0.33 | 0.27 | 0.31 |
| indep_vowel | 29 | 0.31 | 0.24 | 0.28 | 0.24 | 0.17 | 0.28 | 0.31 |
| space | 18 | 0.17 | 0.17 | 0.22 | 0.17 | 0.11 | 0.11 | 0.17 |

**vs CTC control** (adjusted OR of star on disputed units; ratio = OTC/CTC; dAUC with 300-sample cluster bootstrap)

| model | unit | outcome | OR CTC | OR model | ratio (model/CTC) | dAUC(p) | dAUC(-lam) |
|---|---|---|---|---|---|---|---|
| otc_recip | char | p_tok_w0>0.5 | 21.80 [16.29, 29.19] p=2.8e-95 | 18.75 [13.79, 25.51] p=6.5e-78 | 0.86 [0.69, 1.07] p=1.7e-01 | -0.022 [-0.043, +0.001] | -0.027 [-0.048, -0.008] |
| otc_recip | word | p_word_w0>0.5 | 3.00 [1.39, 6.44] p=5.0e-03 | 7.64 [4.72, 12.35] p=1.1e-16 | 2.55 [1.32, 4.94] p=5.6e-03 | +0.009 [-0.000, +0.020] | -0.027 [-0.047, -0.008] |
| otc_recip | bound | ins_w0>0.5 | 70.00 [43.90, 111.61] p=3.0e-71 | 44.54 [29.97, 66.20] p=1.0e-78 | 0.64 [0.43, 0.94] p=2.1e-02 | -0.011 [-0.047, +0.027] | - |
| otc_token | char | p_tok_w0>0.5 | 21.49 [16.03, 28.81] p=1.2e-93 | 19.60 [14.76, 26.03] p=5.1e-94 | 0.91 [0.74, 1.12] p=3.8e-01 | -0.018 [-0.040, +0.001] | -0.032 [-0.052, -0.013] |
| otc_token | word | p_word_w0>0.5 | 3.08 [1.46, 6.51] p=3.2e-03 | 8.02 [4.90, 13.14] p=1.3e-16 | 2.60 [1.37, 4.94] p=3.4e-03 | +0.012 [+0.001, +0.021] | -0.018 [-0.039, +0.003] |
| otc_token | bound | ins_w0>0.5 | 77.02 [48.83, 121.49] p=6.7e-78 | 69.38 [46.66, 103.17] p=2.1e-97 | 0.90 [0.62, 1.30] p=5.8e-01 | -0.025 [-0.061, +0.012] | - |
| otc_token_word | char | p_tok_w0>0.5 | 21.11 [15.72, 28.34] p=2.1e-91 | 18.60 [13.75, 25.18] p=6.0e-80 | 0.88 [0.71, 1.10] p=2.6e-01 | -0.007 [-0.029, +0.017] | -0.014 [-0.033, +0.003] |
| otc_token_word | word | p_word_w0>0.5 | 2.28 [0.94, 5.49] p=6.7e-02 | 3.95 [2.49, 6.25] p=4.7e-09 | 1.73 [0.85, 3.54] p=1.3e-01 | +0.020 [+0.010, +0.031] | +0.029 [+0.013, +0.045] |
| otc_token_word | bound | ins_w0>0.5 | 71.30 [44.45, 114.35] p=4.0e-70 | 67.14 [44.84, 100.55] p=1.1e-92 | 0.94 [0.65, 1.36] p=7.5e-01 | -0.038 [-0.083, +0.007] | - |
| otc_token_word_recip | char | p_tok_w0>0.5 | 17.66 [13.00, 24.01] p=3.5e-75 | 19.70 [14.78, 26.25] p=5.1e-92 | 1.11 [0.89, 1.39] p=3.3e-01 | -0.031 [-0.056, -0.004] | -0.038 [-0.058, -0.015] |
| otc_token_word_recip | word | p_word_w0>0.5 | 2.73 [1.23, 6.05] p=1.3e-02 | 6.28 [3.98, 9.89] p=2.4e-15 | 2.30 [1.22, 4.33] p=9.6e-03 | +0.017 [+0.009, +0.026] | +0.006 [-0.010, +0.022] |
| otc_token_word_recip | bound | ins_w0>0.5 | 82.72 [52.68, 129.91] p=6.0e-82 | 56.99 [39.14, 82.98] p=1.0e-98 | 0.69 [0.49, 0.97] p=3.2e-02 | -0.034 [-0.074, +0.010] | - |
| otc_word | char | p_tok_w0>0.5 | 19.05 [14.06, 25.81] p=1.1e-80 | 19.04 [14.08, 25.75] p=1.2e-81 | 1.00 [0.80, 1.25] p=9.9e-01 | -0.014 [-0.038, +0.012] | -0.015 [-0.034, +0.004] |
| otc_word | word | p_word_w0>0.5 | 2.56 [1.11, 5.92] p=2.7e-02 | 4.72 [2.61, 8.54] p=3.0e-07 | 1.84 [0.97, 3.50] p=6.4e-02 | +0.011 [-0.001, +0.022] | +0.001 [-0.013, +0.016] |
| otc_word | bound | ins_w0>0.5 | 82.28 [52.95, 127.86] p=1.3e-85 | 56.99 [38.87, 83.56] p=3.0e-95 | 0.69 [0.49, 0.98] p=3.6e-02 | +0.024 [-0.006, +0.061] | - |
| otc_word_tuned | char | p_tok_w0>0.5 | 20.33 [15.09, 27.38] p=1.7e-87 | 19.12 [14.12, 25.89] p=3.1e-81 | 0.94 [0.75, 1.18] p=6.0e-01 | -0.006 [-0.028, +0.018] | -0.016 [-0.034, +0.004] |
| otc_word_tuned | word | p_word_w0>0.5 | 1.88 [0.70, 5.05] p=2.1e-01 | 2.73 [1.76, 4.24] p=6.9e-06 | 1.45 [0.62, 3.38] p=3.9e-01 | +0.031 [+0.019, +0.046] | +0.049 [+0.032, +0.067] |
| otc_word_tuned | bound | ins_w0>0.5 | 45.41 [26.12, 78.94] p=1.1e-41 | 123.65 [84.03, 181.97] p=6.4e-132 | 2.72 [1.84, 4.04] p=6.4e-07 | -0.060 [-0.105, -0.018] | - |

## kn (85363 chars, 1263 disputed = 1.48%)

| model | unit | measure | mean(disputed) | mean(undisputed) | ratio | AUC |
|---|---|---|---|---|---|---|
| ctc | char | p_tok_w0 | 0.221 | 0.018 | 12.2 | 0.803 |
| ctc | char | neg_lam | -2.607 | -5.272 | - | 0.844 |
| ctc | word | p_word_w0 | 0.006 | 0.003 | 1.8 | 0.510 |
| ctc | word | neg_lam | -51.367 | -53.657 | - | 0.527 |
| ctc | bound | ins_w0 | 0.134 | 0.012 | 11.0 | 0.769 |
| otc_recip | char | p_tok_w0 | 0.306 | 0.019 | 15.9 | 0.813 |
| otc_recip | char | neg_lam | -1.570 | -6.908 | - | 0.840 |
| otc_recip | char | own_p | 0.298 | 0.017 | 17.5 | 0.813 |
| otc_recip | word | p_word_w0 | 0.010 | 0.004 | 2.5 | 0.514 |
| otc_recip | word | neg_lam | -55.928 | -58.769 | - | 0.530 |
| otc_recip | bound | ins_w0 | 0.230 | 0.022 | 10.4 | 0.757 |
| otc_token | char | p_tok_w0 | 0.289 | 0.018 | 16.1 | 0.816 |
| otc_token | char | neg_lam | -1.657 | -6.750 | - | 0.852 |
| otc_token | char | own_p | 0.288 | 0.017 | 16.9 | 0.816 |
| otc_token | word | p_word_w0 | 0.010 | 0.004 | 2.7 | 0.510 |
| otc_token | word | neg_lam | -55.593 | -58.189 | - | 0.529 |
| otc_token | bound | ins_w0 | 0.204 | 0.020 | 10.0 | 0.759 |
| otc_token_word | char | p_tok_w0 | 0.262 | 0.020 | 13.1 | 0.820 |
| otc_token_word | char | neg_lam | -2.101 | -6.256 | - | 0.847 |
| otc_token_word | char | own_p | 0.254 | 0.017 | 14.7 | 0.818 |
| otc_token_word | word | p_word_w0 | 0.013 | 0.007 | 1.8 | 0.511 |
| otc_token_word | word | neg_lam | -51.324 | -53.837 | - | 0.530 |
| otc_token_word | word | own_p | 0.009 | 0.006 | 1.6 | 0.509 |
| otc_token_word | bound | ins_w0 | 0.172 | 0.018 | 9.3 | 0.756 |
| otc_token_word_recip | char | p_tok_w0 | 0.295 | 0.020 | 15.1 | 0.801 |
| otc_token_word_recip | char | neg_lam | -1.735 | -6.842 | - | 0.823 |
| otc_token_word_recip | char | own_p | 0.283 | 0.016 | 17.5 | 0.796 |
| otc_token_word_recip | word | p_word_w0 | 0.016 | 0.006 | 2.8 | 0.513 |
| otc_token_word_recip | word | neg_lam | -54.874 | -57.791 | - | 0.532 |
| otc_token_word_recip | word | own_p | 0.006 | 0.003 | 2.2 | 0.505 |
| otc_token_word_recip | bound | ins_w0 | 0.234 | 0.022 | 10.8 | 0.744 |
| otc_word | char | p_tok_w0 | 0.220 | 0.017 | 12.9 | 0.762 |
| otc_word | char | neg_lam | -3.040 | -6.092 | - | 0.795 |
| otc_word | word | p_word_w0 | 0.005 | 0.003 | 1.8 | 0.510 |
| otc_word | word | neg_lam | -51.852 | -52.987 | - | 0.517 |
| otc_word | word | own_p | 0.004 | 0.002 | 1.8 | 0.505 |
| otc_word | bound | ins_w0 | 0.207 | 0.018 | 11.8 | 0.803 |
| otc_word_tuned | char | p_tok_w0 | 0.207 | 0.021 | 10.0 | 0.794 |
| otc_word_tuned | char | neg_lam | -2.709 | -5.115 | - | 0.833 |
| otc_word_tuned | word | p_word_w0 | 0.015 | 0.009 | 1.7 | 0.508 |
| otc_word_tuned | word | neg_lam | -48.935 | -51.758 | - | 0.530 |
| otc_word_tuned | word | own_p | 0.016 | 0.010 | 1.5 | 0.515 |
| otc_word_tuned | bound | ins_w0 | 0.143 | 0.012 | 12.3 | 0.759 |

**Share of units with star > 0.5** (disputed / undisputed; and how many of all starred units are disputed)

| model | char: disputed | char: undisputed | char: precision | word (p_word_w0): disputed | word: undisputed | boundary ins (w=0): disputed | boundary: undisputed |
|---|---|---|---|---|---|---|---|
| ctc | 0.206 (260) | 0.0076 (639) | 0.289 | 0.006 | 0.0032 | 0.092 | 0.0013 |
| otc_recip | 0.301 (380) | 0.0132 (1107) | 0.256 | 0.010 | 0.0041 | 0.205 | 0.0039 |
| otc_token | 0.277 (350) | 0.0113 (951) | 0.269 | 0.009 | 0.0037 | 0.172 | 0.0029 |
| otc_token_word | 0.253 (319) | 0.0122 (1026) | 0.237 | 0.013 | 0.0074 | 0.140 | 0.0024 |
| otc_token_word_recip | 0.283 (358) | 0.0131 (1105) | 0.245 | 0.015 | 0.0056 | 0.214 | 0.0036 |
| otc_word | 0.207 (261) | 0.0083 (694) | 0.273 | 0.004 | 0.0027 | 0.163 | 0.0031 |
| otc_word_tuned | 0.192 (243) | 0.0096 (805) | 0.232 | 0.015 | 0.0085 | 0.101 | 0.0015 |

**Disputed chars with star > 0.5, by character class** (token probe, w=0)

| class | n disputed | ctc | otc_recip | otc_token | otc_token_word | otc_token_word_recip | otc_word | otc_word_tuned |
|---|---|---|---|---|---|---|---|---|
| vowel_sign | 453 | 0.27 | 0.37 | 0.36 | 0.34 | 0.37 | 0.27 | 0.25 |
| virama | 327 | 0.15 | 0.30 | 0.26 | 0.21 | 0.27 | 0.13 | 0.13 |
| consonant | 316 | 0.15 | 0.17 | 0.15 | 0.16 | 0.16 | 0.15 | 0.16 |
| space | 138 | 0.22 | 0.35 | 0.30 | 0.26 | 0.30 | 0.25 | 0.19 |
| indep_vowel | 24 | 0.33 | 0.46 | 0.46 | 0.42 | 0.46 | 0.33 | 0.38 |
| other_mark | 5 | 0.40 | 0.40 | 0.40 | 0.40 | 0.60 | 0.60 | 0.40 |

**vs CTC control** (adjusted OR of star on disputed units; ratio = OTC/CTC; dAUC with 300-sample cluster bootstrap)

| model | unit | outcome | OR CTC | OR model | ratio (model/CTC) | dAUC(p) | dAUC(-lam) |
|---|---|---|---|---|---|---|---|
| otc_recip | char | p_tok_w0>0.5 | 21.79 [17.37, 27.33] p=2.2e-156 | 18.93 [15.55, 23.05] p=9.8e-189 | 0.87 [0.75, 1.01] p=5.9e-02 | +0.009 [-0.002, +0.022] | -0.004 [-0.015, +0.009] |
| otc_recip | word | p_word_w0>0.5 | 1.09 [0.47, 2.50] p=8.5e-01 | 1.80 [0.99, 3.27] p=5.4e-02 | 1.66 [0.86, 3.20] p=1.3e-01 | +0.004 [+0.000, +0.008] | +0.003 [-0.006, +0.011] |
| otc_recip | bound | ins_w0>0.5 | 40.83 [28.08, 59.39] p=6.2e-84 | 43.40 [34.12, 55.21] p=4.4e-207 | 1.06 [0.77, 1.48] p=7.2e-01 | -0.012 [-0.029, +0.008] | - |
| otc_token | char | p_tok_w0>0.5 | 18.63 [14.79, 23.48] p=9.2e-136 | 20.80 [16.93, 25.55] p=2.0e-183 | 1.12 [0.96, 1.29] p=1.4e-01 | +0.013 [+0.003, +0.022] | +0.008 [-0.002, +0.019] |
| otc_token | word | p_word_w0>0.5 | 1.05 [0.44, 2.48] p=9.1e-01 | 1.72 [0.91, 3.28] p=9.7e-02 | 1.64 [0.79, 3.40] p=1.8e-01 | -0.000 [-0.004, +0.003] | +0.002 [-0.005, +0.009] |
| otc_token | bound | ins_w0>0.5 | 41.12 [28.35, 59.66] p=2.1e-85 | 46.30 [35.71, 60.03] p=3.7e-184 | 1.13 [0.83, 1.52] p=4.4e-01 | -0.009 [-0.023, +0.002] | - |
| otc_token_word | char | p_tok_w0>0.5 | 18.59 [14.74, 23.45] p=1.9e-134 | 19.77 [16.18, 24.15] p=1.6e-187 | 1.06 [0.92, 1.23] p=4.1e-01 | +0.016 [+0.006, +0.027] | +0.003 [-0.010, +0.013] |
| otc_token_word | word | p_word_w0>0.5 | 0.83 [0.31, 2.23] p=7.1e-01 | 1.02 [0.57, 1.84] p=9.5e-01 | 1.23 [0.55, 2.73] p=6.1e-01 | +0.000 [-0.004, +0.005] | +0.004 [-0.003, +0.010] |
| otc_token_word | bound | ins_w0>0.5 | 42.27 [29.61, 60.33] p=1.6e-94 | 57.23 [43.44, 75.40] p=5.2e-182 | 1.35 [1.01, 1.81] p=4.2e-02 | -0.013 [-0.029, +0.006] | - |
| otc_token_word_recip | char | p_tok_w0>0.5 | 22.25 [17.71, 27.94] p=6.3e-157 | 18.43 [15.12, 22.45] p=1.9e-183 | 0.83 [0.71, 0.97] p=1.8e-02 | -0.002 [-0.014, +0.010] | -0.021 [-0.032, -0.010] |
| otc_token_word_recip | word | p_word_w0>0.5 | 0.92 [0.37, 2.32] p=8.6e-01 | 1.50 [0.87, 2.59] p=1.4e-01 | 1.63 [0.74, 3.59] p=2.3e-01 | +0.003 [-0.001, +0.006] | +0.005 [-0.002, +0.015] |
| otc_token_word_recip | bound | ins_w0>0.5 | 39.93 [27.68, 57.58] p=1.1e-86 | 49.95 [39.75, 62.76] p=3.8e-247 | 1.25 [0.89, 1.75] p=1.9e-01 | -0.025 [-0.041, -0.009] | - |
| otc_word | char | p_tok_w0>0.5 | 19.94 [15.86, 25.08] p=1.8e-144 | 18.63 [14.64, 23.71] p=8.2e-125 | 0.93 [0.79, 1.10] p=4.1e-01 | -0.042 [-0.055, -0.027] | -0.049 [-0.062, -0.034] |
| otc_word | word | p_word_w0>0.5 | 1.04 [0.44, 2.48] p=9.3e-01 | 1.20 [0.57, 2.52] p=6.4e-01 | 1.15 [0.56, 2.37] p=7.0e-01 | -0.001 [-0.004, +0.003] | -0.010 [-0.015, -0.004] |
| otc_word | bound | ins_w0>0.5 | 28.58 [18.74, 43.61] p=1.4e-54 | 34.79 [26.67, 45.36] p=2.6e-151 | 1.22 [0.83, 1.78] p=3.1e-01 | +0.034 [+0.018, +0.053] | - |
| otc_word_tuned | char | p_tok_w0>0.5 | 21.62 [17.16, 27.24] p=6.9e-150 | 18.44 [14.93, 22.78] p=2.6e-161 | 0.85 [0.72, 1.01] p=6.0e-02 | -0.009 [-0.019, +0.002] | -0.011 [-0.022, -0.000] |
| otc_word_tuned | word | p_word_w0>0.5 | 0.87 [0.33, 2.28] p=7.8e-01 | 1.12 [0.66, 1.88] p=6.8e-01 | 1.28 [0.57, 2.87] p=5.5e-01 | -0.002 [-0.006, +0.001] | +0.004 [-0.001, +0.008] |
| otc_word_tuned | bound | ins_w0>0.5 | 36.81 [25.20, 53.78] p=1.3e-77 | 74.05 [52.66, 104.11] p=2.4e-135 | 2.01 [1.45, 2.79] p=2.8e-05 | -0.010 [-0.026, +0.006] | - |

## mg (69788 chars, 1920 disputed = 2.75%)

| model | unit | measure | mean(disputed) | mean(undisputed) | ratio | AUC |
|---|---|---|---|---|---|---|
| ctc | char | p_tok_w0 | 0.168 | 0.019 | 8.7 | 0.856 |
| ctc | char | neg_lam | -2.919 | -5.489 | - | 0.875 |
| ctc | word | p_word_w0 | 0.012 | 0.005 | 2.5 | 0.525 |
| ctc | word | neg_lam | -30.103 | -34.185 | - | 0.580 |
| ctc | bound | ins_w0 | 0.167 | 0.013 | 12.9 | 0.709 |
| otc_recip | char | p_tok_w0 | 0.280 | 0.022 | 12.6 | 0.833 |
| otc_recip | char | neg_lam | -2.127 | -7.342 | - | 0.847 |
| otc_recip | char | own_p | 0.268 | 0.020 | 13.7 | 0.840 |
| otc_recip | word | p_word_w0 | 0.019 | 0.005 | 3.5 | 0.522 |
| otc_recip | word | neg_lam | -37.632 | -41.784 | - | 0.572 |
| otc_recip | bound | ins_w0 | 0.235 | 0.023 | 10.2 | 0.714 |
| otc_token | char | p_tok_w0 | 0.243 | 0.020 | 12.1 | 0.868 |
| otc_token | char | neg_lam | -2.234 | -6.828 | - | 0.879 |
| otc_token | char | own_p | 0.238 | 0.019 | 12.4 | 0.872 |
| otc_token | word | p_word_w0 | 0.017 | 0.005 | 3.5 | 0.522 |
| otc_token | word | neg_lam | -35.926 | -40.169 | - | 0.572 |
| otc_token | bound | ins_w0 | 0.191 | 0.020 | 9.8 | 0.701 |
| otc_token_word | char | p_tok_w0 | 0.221 | 0.021 | 10.6 | 0.865 |
| otc_token_word | char | neg_lam | -2.390 | -6.579 | - | 0.874 |
| otc_token_word | char | own_p | 0.210 | 0.018 | 11.6 | 0.868 |
| otc_token_word | word | p_word_w0 | 0.016 | 0.008 | 2.2 | 0.517 |
| otc_token_word | word | neg_lam | -34.242 | -38.397 | - | 0.576 |
| otc_token_word | word | own_p | 0.008 | 0.005 | 1.7 | 0.512 |
| otc_token_word | bound | ins_w0 | 0.172 | 0.019 | 9.2 | 0.714 |
| otc_token_word_recip | char | p_tok_w0 | 0.285 | 0.023 | 12.4 | 0.836 |
| otc_token_word_recip | char | neg_lam | -1.817 | -7.291 | - | 0.849 |
| otc_token_word_recip | char | own_p | 0.270 | 0.019 | 14.1 | 0.842 |
| otc_token_word_recip | word | p_word_w0 | 0.022 | 0.007 | 3.3 | 0.523 |
| otc_token_word_recip | word | neg_lam | -36.665 | -41.382 | - | 0.580 |
| otc_token_word_recip | word | own_p | 0.003 | 0.003 | 1.2 | 0.505 |
| otc_token_word_recip | bound | ins_w0 | 0.249 | 0.023 | 10.8 | 0.714 |
| otc_word | char | p_tok_w0 | 0.189 | 0.019 | 10.1 | 0.699 |
| otc_word | char | neg_lam | -4.006 | -6.564 | - | 0.734 |
| otc_word | word | p_word_w0 | 0.008 | 0.004 | 2.1 | 0.516 |
| otc_word | word | neg_lam | -33.758 | -36.613 | - | 0.564 |
| otc_word | word | own_p | 0.009 | 0.004 | 2.1 | 0.510 |
| otc_word | bound | ins_w0 | 0.253 | 0.021 | 11.9 | 0.753 |
| otc_word_tuned | char | p_tok_w0 | 0.173 | 0.020 | 8.7 | 0.851 |
| otc_word_tuned | char | neg_lam | -2.889 | -5.521 | - | 0.873 |
| otc_word_tuned | word | p_word_w0 | 0.014 | 0.006 | 2.3 | 0.522 |
| otc_word_tuned | word | neg_lam | -30.701 | -35.239 | - | 0.586 |
| otc_word_tuned | word | own_p | 0.020 | 0.007 | 2.9 | 0.533 |
| otc_word_tuned | bound | ins_w0 | 0.145 | 0.013 | 10.9 | 0.706 |

**Share of units with star > 0.5** (disputed / undisputed; and how many of all starred units are disputed)

| model | char: disputed | char: undisputed | char: precision | word (p_word_w0): disputed | word: undisputed | boundary ins (w=0): disputed | boundary: undisputed |
|---|---|---|---|---|---|---|---|
| ctc | 0.141 (270) | 0.0092 (627) | 0.301 | 0.013 | 0.0046 | 0.121 | 0.0018 |
| otc_recip | 0.268 (515) | 0.0170 (1157) | 0.308 | 0.019 | 0.0054 | 0.207 | 0.0050 |
| otc_token | 0.230 (442) | 0.0135 (913) | 0.326 | 0.016 | 0.0049 | 0.143 | 0.0032 |
| otc_token_word | 0.201 (386) | 0.0139 (943) | 0.290 | 0.015 | 0.0070 | 0.122 | 0.0026 |
| otc_token_word_recip | 0.274 (526) | 0.0178 (1209) | 0.303 | 0.021 | 0.0066 | 0.222 | 0.0053 |
| otc_word | 0.174 (335) | 0.0109 (738) | 0.312 | 0.007 | 0.0038 | 0.221 | 0.0053 |
| otc_word_tuned | 0.149 (286) | 0.0099 (675) | 0.298 | 0.013 | 0.0056 | 0.109 | 0.0018 |

**Disputed chars with star > 0.5, by character class** (token probe, w=0)

| class | n disputed | ctc | otc_recip | otc_token | otc_token_word | otc_token_word_recip | otc_word | otc_word_tuned |
|---|---|---|---|---|---|---|---|---|
| indep_vowel | 718 | 0.08 | 0.20 | 0.14 | 0.11 | 0.19 | 0.12 | 0.08 |
| vowel_sign | 517 | 0.18 | 0.32 | 0.30 | 0.25 | 0.33 | 0.22 | 0.20 |
| consonant | 494 | 0.14 | 0.27 | 0.22 | 0.22 | 0.28 | 0.17 | 0.15 |
| space | 71 | 0.23 | 0.38 | 0.34 | 0.28 | 0.31 | 0.25 | 0.27 |
| virama | 71 | 0.27 | 0.35 | 0.52 | 0.44 | 0.46 | 0.24 | 0.25 |
| other_mark | 49 | 0.27 | 0.47 | 0.43 | 0.35 | 0.45 | 0.27 | 0.24 |

**vs CTC control** (adjusted OR of star on disputed units; ratio = OTC/CTC; dAUC with 300-sample cluster bootstrap)

| model | unit | outcome | OR CTC | OR model | ratio (model/CTC) | dAUC(p) | dAUC(-lam) |
|---|---|---|---|---|---|---|---|
| otc_recip | char | p_tok_w0>0.5 | 11.11 [8.86, 13.93] p=1.5e-96 | 14.89 [12.52, 17.71] p=4.5e-205 | 1.34 [1.12, 1.60] p=1.1e-03 | -0.022 [-0.031, -0.014] | -0.028 [-0.036, -0.019] |
| otc_recip | word | p_word_w0>0.5 | 1.69 [0.96, 2.96] p=6.7e-02 | 2.92 [1.84, 4.64] p=5.1e-06 | 1.73 [1.09, 2.76] p=2.1e-02 | -0.002 [-0.007, +0.002] | -0.008 [-0.016, -0.000] |
| otc_recip | bound | ins_w0>0.5 | 56.56 [39.56, 80.85] p=1.5e-108 | 48.73 [36.95, 64.25] p=5.3e-167 | 0.86 [0.63, 1.17] p=3.5e-01 | +0.005 [-0.017, +0.029] | - |
| otc_token | char | p_tok_w0>0.5 | 11.09 [8.83, 13.92] p=2.0e-95 | 16.44 [13.46, 20.07] p=3.3e-166 | 1.48 [1.24, 1.77] p=1.2e-05 | +0.012 [+0.005, +0.019] | +0.004 [-0.002, +0.011] |
| otc_token | word | p_word_w0>0.5 | 1.75 [1.03, 2.99] p=3.9e-02 | 3.15 [1.94, 5.12] p=3.8e-06 | 1.80 [1.13, 2.85] p=1.3e-02 | -0.003 [-0.007, +0.002] | -0.008 [-0.014, -0.002] |
| otc_token | bound | ins_w0>0.5 | 51.76 [36.52, 73.37] p=6.3e-109 | 63.32 [45.69, 87.75] p=4.7e-137 | 1.22 [0.93, 1.62] p=1.6e-01 | -0.008 [-0.028, +0.012] | - |
| otc_token_word | char | p_tok_w0>0.5 | 11.36 [9.05, 14.25] p=7.2e-98 | 13.39 [10.94, 16.38] p=3.6e-140 | 1.18 [0.98, 1.42] p=8.6e-02 | +0.009 [+0.002, +0.016] | -0.002 [-0.007, +0.004] |
| otc_token_word | word | p_word_w0>0.5 | 1.50 [0.83, 2.71] p=1.8e-01 | 1.78 [1.14, 2.77] p=1.1e-02 | 1.18 [0.73, 1.92] p=4.9e-01 | -0.008 [-0.013, -0.004] | -0.004 [-0.009, +0.002] |
| otc_token_word | bound | ins_w0>0.5 | 57.73 [40.95, 81.39] p=1.5e-118 | 56.89 [40.40, 80.10] p=1.6e-118 | 0.99 [0.74, 1.31] p=9.2e-01 | +0.005 [-0.013, +0.020] | - |
| otc_token_word_recip | char | p_tok_w0>0.5 | 10.69 [8.50, 13.45] p=2.4e-91 | 14.15 [11.87, 16.87] p=6.0e-192 | 1.32 [1.10, 1.60] p=3.5e-03 | -0.019 [-0.028, -0.011] | -0.026 [-0.035, -0.018] |
| otc_token_word_recip | word | p_word_w0>0.5 | 1.60 [0.90, 2.83] p=1.1e-01 | 2.51 [1.65, 3.81] p=1.6e-05 | 1.57 [0.95, 2.58] p=7.7e-02 | -0.001 [-0.008, +0.004] | +0.000 [-0.007, +0.007] |
| otc_token_word_recip | bound | ins_w0>0.5 | 59.00 [41.43, 84.01] p=3.4e-113 | 52.96 [40.71, 68.88] p=1.8e-192 | 0.90 [0.67, 1.21] p=4.8e-01 | +0.005 [-0.020, +0.026] | - |
| otc_word | char | p_tok_w0>0.5 | 11.54 [9.21, 14.46] p=3.8e-100 | 12.63 [10.33, 15.45] p=1.3e-134 | 1.09 [0.92, 1.30] p=3.1e-01 | -0.156 [-0.172, -0.142] | -0.141 [-0.156, -0.128] |
| otc_word | word | p_word_w0>0.5 | 1.81 [1.04, 3.16] p=3.6e-02 | 1.34 [0.72, 2.47] p=3.6e-01 | 0.74 [0.42, 1.28] p=2.8e-01 | -0.008 [-0.014, -0.004] | -0.016 [-0.024, -0.010] |
| otc_word | bound | ins_w0>0.5 | 66.26 [46.33, 94.78] p=1.1e-116 | 40.22 [30.07, 53.81] p=9.5e-137 | 0.61 [0.45, 0.83] p=1.6e-03 | +0.044 [+0.024, +0.064] | - |
| otc_word_tuned | char | p_tok_w0>0.5 | 11.19 [8.89, 14.08] p=2.6e-94 | 12.48 [9.94, 15.66] p=3.1e-105 | 1.12 [0.94, 1.32] p=2.0e-01 | -0.005 [-0.012, +0.002] | -0.002 [-0.010, +0.005] |
| otc_word_tuned | word | p_word_w0>0.5 | 1.60 [0.90, 2.84] p=1.1e-01 | 1.84 [1.11, 3.06] p=1.9e-02 | 1.15 [0.71, 1.87] p=5.7e-01 | -0.002 [-0.007, +0.002] | +0.006 [+0.003, +0.010] |
| otc_word_tuned | bound | ins_w0>0.5 | 46.07 [31.60, 67.17] p=3.5e-88 | 74.98 [50.11, 112.21] p=7.7e-98 | 1.63 [1.18, 2.24] p=2.9e-03 | -0.003 [-0.021, +0.014] | - |

## mr (70129 chars, 237 disputed = 0.34%)

| model | unit | measure | mean(disputed) | mean(undisputed) | ratio | AUC |
|---|---|---|---|---|---|---|
| ctc | char | p_tok_w0 | 0.304 | 0.018 | 16.5 | 0.823 |
| ctc | char | neg_lam | -2.110 | -5.741 | - | 0.828 |
| ctc | word | p_word_w0 | 0.013 | 0.004 | 3.5 | 0.505 |
| ctc | word | neg_lam | -60.382 | -47.525 | - | 0.362 |
| ctc | bound | ins_w0 | 0.161 | 0.015 | 10.6 | 0.771 |
| otc_recip | char | p_tok_w0 | 0.350 | 0.020 | 17.8 | 0.789 |
| otc_recip | char | neg_lam | -1.557 | -7.184 | - | 0.804 |
| otc_recip | char | own_p | 0.328 | 0.018 | 18.4 | 0.788 |
| otc_recip | word | p_word_w0 | 0.017 | 0.005 | 3.3 | 0.507 |
| otc_recip | word | neg_lam | -65.306 | -50.751 | - | 0.357 |
| otc_recip | bound | ins_w0 | 0.241 | 0.022 | 10.7 | 0.755 |
| otc_token | char | p_tok_w0 | 0.312 | 0.019 | 16.4 | 0.779 |
| otc_token | char | neg_lam | -1.965 | -7.023 | - | 0.799 |
| otc_token | char | own_p | 0.297 | 0.018 | 16.3 | 0.778 |
| otc_token | word | p_word_w0 | 0.020 | 0.004 | 4.6 | 0.510 |
| otc_token | word | neg_lam | -64.214 | -50.284 | - | 0.357 |
| otc_token | bound | ins_w0 | 0.242 | 0.021 | 11.4 | 0.774 |
| otc_token_word | char | p_tok_w0 | 0.337 | 0.022 | 15.1 | 0.806 |
| otc_token_word | char | neg_lam | -1.322 | -6.656 | - | 0.830 |
| otc_token_word | char | own_p | 0.300 | 0.019 | 16.2 | 0.782 |
| otc_token_word | word | p_word_w0 | 0.031 | 0.011 | 2.9 | 0.518 |
| otc_token_word | word | neg_lam | -58.566 | -46.680 | - | 0.365 |
| otc_token_word | word | own_p | 0.016 | 0.009 | 1.9 | 0.508 |
| otc_token_word | bound | ins_w0 | 0.215 | 0.020 | 10.8 | 0.774 |
| otc_token_word_recip | char | p_tok_w0 | 0.359 | 0.019 | 18.4 | 0.811 |
| otc_token_word_recip | char | neg_lam | -1.003 | -7.101 | - | 0.821 |
| otc_token_word_recip | char | own_p | 0.328 | 0.017 | 19.6 | 0.790 |
| otc_token_word_recip | word | p_word_w0 | 0.013 | 0.005 | 2.4 | 0.512 |
| otc_token_word_recip | word | neg_lam | -63.004 | -49.287 | - | 0.354 |
| otc_token_word_recip | word | own_p | 0.006 | 0.002 | 2.7 | 0.502 |
| otc_token_word_recip | bound | ins_w0 | 0.238 | 0.022 | 10.9 | 0.767 |
| otc_word | char | p_tok_w0 | 0.291 | 0.018 | 15.8 | 0.773 |
| otc_word | char | neg_lam | -2.632 | -6.400 | - | 0.784 |
| otc_word | word | p_word_w0 | 0.012 | 0.004 | 3.3 | 0.510 |
| otc_word | word | neg_lam | -57.957 | -44.202 | - | 0.345 |
| otc_word | word | own_p | 0.017 | 0.004 | 4.8 | 0.514 |
| otc_word | bound | ins_w0 | 0.178 | 0.018 | 9.8 | 0.746 |
| otc_word_tuned | char | p_tok_w0 | 0.276 | 0.026 | 10.4 | 0.796 |
| otc_word_tuned | char | neg_lam | -2.270 | -5.671 | - | 0.818 |
| otc_word_tuned | word | p_word_w0 | 0.048 | 0.023 | 2.0 | 0.516 |
| otc_word_tuned | word | neg_lam | -53.704 | -44.058 | - | 0.389 |
| otc_word_tuned | word | own_p | 0.058 | 0.025 | 2.4 | 0.525 |
| otc_word_tuned | bound | ins_w0 | 0.146 | 0.015 | 9.9 | 0.707 |

**Share of units with star > 0.5** (disputed / undisputed; and how many of all starred units are disputed)

| model | char: disputed | char: undisputed | char: precision | word (p_word_w0): disputed | word: undisputed | boundary ins (w=0): disputed | boundary: undisputed |
|---|---|---|---|---|---|---|---|
| ctc | 0.283 (67) | 0.0099 (694) | 0.088 | 0.012 | 0.0035 | 0.116 | 0.0023 |
| otc_recip | 0.350 (83) | 0.0146 (1018) | 0.075 | 0.018 | 0.0054 | 0.197 | 0.0046 |
| otc_token | 0.304 (72) | 0.0137 (960) | 0.070 | 0.021 | 0.0043 | 0.190 | 0.0038 |
| otc_token_word | 0.338 (80) | 0.0161 (1127) | 0.066 | 0.030 | 0.0106 | 0.177 | 0.0032 |
| otc_token_word_recip | 0.350 (83) | 0.0140 (977) | 0.078 | 0.012 | 0.0053 | 0.211 | 0.0040 |
| otc_word | 0.287 (68) | 0.0105 (737) | 0.084 | 0.012 | 0.0037 | 0.156 | 0.0041 |
| otc_word_tuned | 0.253 (60) | 0.0173 (1210) | 0.047 | 0.048 | 0.0230 | 0.116 | 0.0022 |

**Disputed chars with star > 0.5, by character class** (token probe, w=0)

| class | n disputed | ctc | otc_recip | otc_token | otc_token_word | otc_token_word_recip | otc_word | otc_word_tuned |
|---|---|---|---|---|---|---|---|---|
| consonant | 89 | 0.27 | 0.26 | 0.33 | 0.33 | 0.35 | 0.27 | 0.30 |
| vowel_sign | 72 | 0.35 | 0.47 | 0.36 | 0.36 | 0.39 | 0.35 | 0.28 |
| other_mark | 33 | 0.30 | 0.45 | 0.27 | 0.45 | 0.27 | 0.30 | 0.24 |
| space | 20 | 0.05 | 0.10 | 0.05 | 0.05 | 0.05 | 0.10 | 0.00 |
| virama | 12 | 0.33 | 0.67 | 0.33 | 0.42 | 0.58 | 0.42 | 0.17 |
| indep_vowel | 11 | 0.27 | 0.09 | 0.27 | 0.36 | 0.64 | 0.18 | 0.27 |

**vs CTC control** (adjusted OR of star on disputed units; ratio = OTC/CTC; dAUC with 300-sample cluster bootstrap)

| model | unit | outcome | OR CTC | OR model | ratio (model/CTC) | dAUC(p) | dAUC(-lam) |
|---|---|---|---|---|---|---|---|
| otc_recip | char | p_tok_w0>0.5 | 21.50 [13.62, 33.94] p=1.3e-39 | 21.55 [14.64, 31.71] p=1.1e-54 | 1.00 [0.73, 1.37] p=9.9e-01 | -0.034 [-0.059, -0.011] | -0.024 [-0.052, +0.001] |
| otc_recip | word | p_word_w0>0.5 | 2.03 [0.64, 6.45] p=2.3e-01 | 3.68 [1.32, 10.29] p=1.3e-02 | 1.82 [0.59, 5.56] p=3.0e-01 | +0.003 [-0.005, +0.009] | -0.006 [-0.022, +0.008] |
| otc_recip | bound | ins_w0>0.5 | 12.86 [3.04, 54.36] p=5.2e-04 | 54.41 [34.80, 85.07] p=8.9e-69 | 4.23 [1.24, 14.44] p=2.1e-02 | -0.016 [-0.056, +0.024] | - |
| otc_token | char | p_tok_w0>0.5 | 21.73 [13.84, 34.12] p=9.1e-41 | 18.81 [11.41, 31.02] p=1.4e-30 | 0.87 [0.60, 1.25] p=4.4e-01 | -0.044 [-0.070, -0.018] | -0.029 [-0.056, -0.003] |
| otc_token | word | p_word_w0>0.5 | 2.35 [0.80, 6.91] p=1.2e-01 | 5.46 [2.24, 13.28] p=1.8e-04 | 2.32 [0.94, 5.76] p=6.9e-02 | +0.005 [-0.002, +0.011] | -0.005 [-0.018, +0.008] |
| otc_token | bound | ins_w0>0.5 | 12.98 [3.26, 51.66] p=2.8e-04 | 56.71 [35.86, 89.69] p=8.6e-67 | 4.37 [1.35, 14.09] p=1.4e-02 | +0.003 [-0.036, +0.046] | - |
| otc_token_word | char | p_tok_w0>0.5 | 21.88 [13.97, 34.28] p=2.2e-41 | 22.87 [15.63, 33.44] p=1.5e-58 | 1.04 [0.78, 1.40] p=7.7e-01 | -0.017 [-0.049, +0.010] | +0.002 [-0.017, +0.025] |
| otc_token_word | word | p_word_w0>0.5 | 1.39 [0.31, 6.18] p=6.7e-01 | 1.80 [0.74, 4.36] p=1.9e-01 | 1.30 [0.35, 4.86] p=7.0e-01 | +0.013 [+0.004, +0.022] | +0.002 [-0.011, +0.015] |
| otc_token_word | bound | ins_w0>0.5 | 17.07 [5.60, 51.98] p=5.9e-07 | 85.51 [56.70, 128.95] p=5.8e-100 | 5.01 [2.08, 12.06] p=3.2e-04 | +0.003 [-0.035, +0.047] | - |
| otc_token_word_recip | char | p_tok_w0>0.5 | 17.58 [10.83, 28.55] p=4.8e-31 | 27.71 [17.72, 43.33] p=5.2e-48 | 1.58 [1.06, 2.34] p=2.4e-02 | -0.012 [-0.030, +0.007] | -0.007 [-0.024, +0.009] |
| otc_token_word_recip | word | p_word_w0>0.5 | 1.64 [0.43, 6.21] p=4.7e-01 | 2.18 [0.82, 5.81] p=1.2e-01 | 1.33 [0.49, 3.60] p=5.8e-01 | +0.007 [-0.002, +0.016] | -0.008 [-0.023, +0.005] |
| otc_token_word_recip | bound | ins_w0>0.5 | 13.03 [3.20, 53.05] p=3.4e-04 | 60.69 [39.40, 93.47] p=1.8e-77 | 4.66 [1.45, 14.97] p=9.8e-03 | -0.004 [-0.039, +0.032] | - |
| otc_word | char | p_tok_w0>0.5 | 18.79 [11.69, 30.19] p=8.5e-34 | 24.58 [15.28, 39.55] p=9.1e-40 | 1.31 [0.90, 1.89] p=1.5e-01 | -0.050 [-0.078, -0.021] | -0.044 [-0.082, -0.017] |
| otc_word | word | p_word_w0>0.5 | 1.66 [0.43, 6.33] p=4.6e-01 | 2.26 [0.63, 8.03] p=2.1e-01 | 1.36 [0.41, 4.49] p=6.1e-01 | +0.006 [-0.002, +0.013] | -0.017 [-0.029, -0.005] |
| otc_word | bound | ins_w0>0.5 | 15.35 [4.22, 55.85] p=3.4e-05 | 60.53 [38.56, 95.01] p=3.7e-71 | 3.94 [1.42, 10.96] p=8.5e-03 | -0.025 [-0.067, +0.015] | - |
| otc_word_tuned | char | p_tok_w0>0.5 | 21.62 [13.79, 33.91] p=6.5e-41 | 16.62 [9.72, 28.42] p=9.1e-25 | 0.77 [0.53, 1.12] p=1.7e-01 | -0.027 [-0.057, -0.002] | -0.010 [-0.034, +0.013] |
| otc_word_tuned | word | p_word_w0>0.5 | 1.03 [0.18, 5.92] p=9.7e-01 | 1.02 [0.49, 2.13] p=9.5e-01 | 0.99 [0.20, 4.90] p=9.9e-01 | +0.011 [-0.000, +0.025] | +0.026 [+0.009, +0.043] |
| otc_word_tuned | bound | ins_w0>0.5 | 18.71 [6.48, 54.05] p=6.2e-08 | 68.10 [42.61, 108.83] p=1.1e-69 | 3.64 [1.64, 8.09] p=1.5e-03 | -0.064 [-0.121, -0.011] | - |

## mt (68450 chars, 2055 disputed = 3.00%)

| model | unit | measure | mean(disputed) | mean(undisputed) | ratio | AUC |
|---|---|---|---|---|---|---|
| ctc | char | p_tok_w0 | 0.171 | 0.017 | 9.9 | 0.788 |
| ctc | char | neg_lam | -3.274 | -5.550 | - | 0.801 |
| ctc | word | p_word_w0 | 0.009 | 0.003 | 3.0 | 0.525 |
| ctc | word | neg_lam | -31.387 | -38.533 | - | 0.634 |
| ctc | bound | ins_w0 | 0.118 | 0.013 | 9.0 | 0.671 |
| otc_recip | char | p_tok_w0 | 0.245 | 0.020 | 12.0 | 0.770 |
| otc_recip | char | neg_lam | -2.770 | -7.657 | - | 0.777 |
| otc_recip | char | own_p | 0.239 | 0.018 | 12.9 | 0.772 |
| otc_recip | word | p_word_w0 | 0.026 | 0.004 | 6.3 | 0.531 |
| otc_recip | word | neg_lam | -36.833 | -44.513 | - | 0.627 |
| otc_recip | bound | ins_w0 | 0.176 | 0.024 | 7.3 | 0.654 |
| otc_token | char | p_tok_w0 | 0.228 | 0.018 | 12.3 | 0.800 |
| otc_token | char | neg_lam | -2.811 | -7.282 | - | 0.802 |
| otc_token | char | own_p | 0.223 | 0.017 | 12.8 | 0.804 |
| otc_token | word | p_word_w0 | 0.017 | 0.003 | 5.4 | 0.529 |
| otc_token | word | neg_lam | -36.003 | -43.417 | - | 0.627 |
| otc_token | bound | ins_w0 | 0.169 | 0.021 | 7.9 | 0.662 |
| otc_token_word | char | p_tok_w0 | 0.220 | 0.018 | 12.1 | 0.806 |
| otc_token_word | char | neg_lam | -2.875 | -7.109 | - | 0.812 |
| otc_token_word | char | own_p | 0.209 | 0.016 | 12.7 | 0.803 |
| otc_token_word | word | p_word_w0 | 0.016 | 0.004 | 3.7 | 0.525 |
| otc_token_word | word | neg_lam | -34.748 | -42.239 | - | 0.631 |
| otc_token_word | word | own_p | 0.007 | 0.002 | 3.2 | 0.514 |
| otc_token_word | bound | ins_w0 | 0.138 | 0.021 | 6.6 | 0.653 |
| otc_token_word_recip | char | p_tok_w0 | 0.235 | 0.020 | 11.7 | 0.774 |
| otc_token_word_recip | char | neg_lam | -2.956 | -7.541 | - | 0.782 |
| otc_token_word_recip | char | own_p | 0.222 | 0.018 | 12.3 | 0.769 |
| otc_token_word_recip | word | p_word_w0 | 0.020 | 0.004 | 4.9 | 0.523 |
| otc_token_word_recip | word | neg_lam | -35.841 | -43.277 | - | 0.632 |
| otc_token_word_recip | word | own_p | 0.004 | 0.001 | 4.4 | 0.506 |
| otc_token_word_recip | bound | ins_w0 | 0.170 | 0.024 | 7.2 | 0.652 |
| otc_word | char | p_tok_w0 | 0.179 | 0.017 | 10.7 | 0.679 |
| otc_word | char | neg_lam | -4.179 | -6.740 | - | 0.708 |
| otc_word | word | p_word_w0 | 0.011 | 0.003 | 3.7 | 0.519 |
| otc_word | word | neg_lam | -32.512 | -38.293 | - | 0.632 |
| otc_word | word | own_p | 0.011 | 0.003 | 3.8 | 0.517 |
| otc_word | bound | ins_w0 | 0.185 | 0.022 | 8.6 | 0.767 |
| otc_word_tuned | char | p_tok_w0 | 0.174 | 0.018 | 9.8 | 0.781 |
| otc_word_tuned | char | neg_lam | -3.246 | -5.671 | - | 0.799 |
| otc_word_tuned | word | p_word_w0 | 0.010 | 0.003 | 3.0 | 0.523 |
| otc_word_tuned | word | neg_lam | -32.533 | -40.107 | - | 0.636 |
| otc_word_tuned | word | own_p | 0.016 | 0.005 | 3.5 | 0.542 |
| otc_word_tuned | bound | ins_w0 | 0.127 | 0.014 | 9.3 | 0.681 |

**Share of units with star > 0.5** (disputed / undisputed; and how many of all starred units are disputed)

| model | char: disputed | char: undisputed | char: precision | word (p_word_w0): disputed | word: undisputed | boundary ins (w=0): disputed | boundary: undisputed |
|---|---|---|---|---|---|---|---|
| ctc | 0.155 (319) | 0.0078 (521) | 0.380 | 0.009 | 0.0032 | 0.094 | 0.0019 |
| otc_recip | 0.242 (497) | 0.0160 (1063) | 0.319 | 0.025 | 0.0038 | 0.149 | 0.0052 |
| otc_token | 0.218 (447) | 0.0129 (858) | 0.343 | 0.016 | 0.0031 | 0.141 | 0.0039 |
| otc_token_word | 0.209 (429) | 0.0126 (835) | 0.339 | 0.014 | 0.0040 | 0.112 | 0.0035 |
| otc_token_word_recip | 0.226 (465) | 0.0154 (1020) | 0.313 | 0.020 | 0.0042 | 0.147 | 0.0050 |
| otc_word | 0.167 (344) | 0.0094 (624) | 0.355 | 0.011 | 0.0030 | 0.169 | 0.0058 |
| otc_word_tuned | 0.158 (324) | 0.0084 (555) | 0.369 | 0.010 | 0.0033 | 0.101 | 0.0020 |

**Disputed chars with star > 0.5, by character class** (token probe, w=0)

| class | n disputed | ctc | otc_recip | otc_token | otc_token_word | otc_token_word_recip | otc_word | otc_word_tuned |
|---|---|---|---|---|---|---|---|---|
| vowel_sign | 668 | 0.16 | 0.22 | 0.20 | 0.20 | 0.20 | 0.16 | 0.16 |
| consonant | 595 | 0.22 | 0.28 | 0.26 | 0.24 | 0.27 | 0.20 | 0.21 |
| indep_vowel | 398 | 0.09 | 0.20 | 0.14 | 0.15 | 0.15 | 0.12 | 0.10 |
| other_mark | 276 | 0.10 | 0.24 | 0.25 | 0.22 | 0.28 | 0.18 | 0.12 |
| space | 60 | 0.22 | 0.23 | 0.23 | 0.22 | 0.25 | 0.15 | 0.18 |
| virama | 58 | 0.17 | 0.41 | 0.29 | 0.29 | 0.31 | 0.19 | 0.19 |

**vs CTC control** (adjusted OR of star on disputed units; ratio = OTC/CTC; dAUC with 300-sample cluster bootstrap)

| model | unit | outcome | OR CTC | OR model | ratio (model/CTC) | dAUC(p) | dAUC(-lam) |
|---|---|---|---|---|---|---|---|
| otc_recip | char | p_tok_w0>0.5 | 13.59 [10.74, 17.19] p=5.1e-105 | 10.64 [8.99, 12.60] p=7.3e-166 | 0.78 [0.66, 0.93] p=5.7e-03 | -0.018 [-0.028, -0.007] | -0.024 [-0.035, -0.014] |
| otc_recip | word | p_word_w0>0.5 | 1.68 [0.93, 3.06] p=8.7e-02 | 5.33 [3.58, 7.94] p=1.7e-16 | 3.17 [1.80, 5.58] p=6.7e-05 | +0.005 [-0.001, +0.011] | -0.007 [-0.013, -0.001] |
| otc_recip | bound | ins_w0>0.5 | 37.62 [27.17, 52.07] p=5.3e-106 | 39.07 [30.47, 50.09] p=1.4e-183 | 1.04 [0.79, 1.37] p=7.9e-01 | -0.017 [-0.034, +0.001] | - |
| otc_token | char | p_tok_w0>0.5 | 13.98 [11.05, 17.68] p=2.7e-107 | 11.47 [9.41, 13.99] p=3.2e-128 | 0.82 [0.70, 0.97] p=1.7e-02 | +0.012 [+0.005, +0.020] | +0.001 [-0.008, +0.009] |
| otc_token | word | p_word_w0>0.5 | 1.72 [0.96, 3.10] p=7.0e-02 | 4.55 [2.89, 7.14] p=5.0e-11 | 2.64 [1.58, 4.41] p=2.1e-04 | +0.004 [-0.000, +0.009] | -0.006 [-0.013, -0.001] |
| otc_token | bound | ins_w0>0.5 | 44.10 [31.75, 61.24] p=5.2e-113 | 46.49 [35.40, 61.05] p=7.1e-168 | 1.05 [0.80, 1.40] p=7.1e-01 | -0.009 [-0.025, +0.008] | - |
| otc_token_word | char | p_tok_w0>0.5 | 13.88 [10.98, 17.54] p=1.8e-107 | 10.93 [8.43, 14.19] p=2.5e-72 | 0.79 [0.66, 0.94] p=7.0e-03 | +0.018 [+0.009, +0.026] | +0.011 [+0.003, +0.019] |
| otc_token_word | word | p_word_w0>0.5 | 1.60 [0.87, 2.95] p=1.3e-01 | 2.72 [1.69, 4.37] p=3.8e-05 | 1.70 [0.99, 2.90] p=5.4e-02 | +0.000 [-0.004, +0.004] | -0.003 [-0.008, +0.003] |
| otc_token_word | bound | ins_w0>0.5 | 42.92 [30.94, 59.53] p=2.8e-112 | 39.45 [29.22, 53.26] p=3.0e-127 | 0.92 [0.70, 1.20] p=5.4e-01 | -0.019 [-0.036, -0.002] | - |
| otc_token_word_recip | char | p_tok_w0>0.5 | 13.10 [10.34, 16.60] p=1.4e-100 | 9.56 [7.84, 11.66] p=5.2e-110 | 0.73 [0.63, 0.85] p=7.1e-05 | -0.014 [-0.024, -0.005] | -0.019 [-0.029, -0.009] |
| otc_token_word_recip | word | p_word_w0>0.5 | 1.65 [0.90, 3.02] p=1.1e-01 | 3.62 [2.42, 5.40] p=3.5e-10 | 2.19 [1.28, 3.75] p=4.1e-03 | -0.002 [-0.007, +0.002] | -0.002 [-0.007, +0.003] |
| otc_token_word_recip | bound | ins_w0>0.5 | 39.08 [28.46, 53.66] p=1.3e-113 | 39.89 [30.71, 51.82] p=7.2e-168 | 1.02 [0.76, 1.37] p=8.9e-01 | -0.019 [-0.037, +0.002] | - |
| otc_word | char | p_tok_w0>0.5 | 13.87 [10.91, 17.64] p=4.8e-102 | 11.27 [8.74, 14.53] p=5.1e-78 | 0.81 [0.69, 0.95] p=1.2e-02 | -0.109 [-0.123, -0.095] | -0.092 [-0.105, -0.079] |
| otc_word | word | p_word_w0>0.5 | 1.75 [0.97, 3.17] p=6.4e-02 | 2.77 [1.60, 4.77] p=2.5e-04 | 1.58 [0.92, 2.72] p=1.0e-01 | -0.006 [-0.011, -0.001] | -0.002 [-0.007, +0.003] |
| otc_word | bound | ins_w0>0.5 | 31.63 [22.32, 44.82] p=4.8e-84 | 21.55 [16.77, 27.68] p=1.8e-127 | 0.68 [0.51, 0.92] p=1.1e-02 | +0.096 [+0.074, +0.115] | - |
| otc_word_tuned | char | p_tok_w0>0.5 | 14.38 [11.31, 18.29] p=4.6e-105 | 13.67 [10.56, 17.70] p=9.4e-88 | 0.95 [0.81, 1.12] p=5.4e-01 | -0.007 [-0.016, -0.000] | -0.001 [-0.008, +0.006] |
| otc_word_tuned | word | p_word_w0>0.5 | 1.71 [0.95, 3.08] p=7.3e-02 | 2.56 [1.53, 4.28] p=3.6e-04 | 1.49 [0.89, 2.52] p=1.3e-01 | -0.002 [-0.007, +0.003] | +0.002 [-0.002, +0.005] |
| otc_word_tuned | bound | ins_w0>0.5 | 48.64 [34.95, 67.69] p=2.0e-117 | 50.55 [35.73, 71.50] p=6.6e-109 | 1.04 [0.81, 1.34] p=7.7e-01 | +0.009 [-0.003, +0.023] | - |

## te (81154 chars, 540 disputed = 0.67%)

| model | unit | measure | mean(disputed) | mean(undisputed) | ratio | AUC |
|---|---|---|---|---|---|---|
| ctc | char | p_tok_w0 | 0.255 | 0.018 | 14.3 | 0.840 |
| ctc | char | neg_lam | -2.068 | -5.259 | - | 0.863 |
| ctc | word | p_word_w0 | 0.008 | 0.003 | 2.7 | 0.509 |
| ctc | word | neg_lam | -57.791 | -51.614 | - | 0.429 |
| ctc | bound | ins_w0 | 0.159 | 0.013 | 12.6 | 0.708 |
| otc_recip | char | p_tok_w0 | 0.327 | 0.018 | 17.9 | 0.835 |
| otc_recip | char | neg_lam | -1.101 | -6.595 | - | 0.849 |
| otc_recip | char | own_p | 0.317 | 0.017 | 19.1 | 0.836 |
| otc_recip | word | p_word_w0 | 0.011 | 0.004 | 2.6 | 0.510 |
| otc_recip | word | neg_lam | -62.805 | -56.582 | - | 0.440 |
| otc_recip | bound | ins_w0 | 0.227 | 0.021 | 11.0 | 0.720 |
| otc_token | char | p_tok_w0 | 0.301 | 0.018 | 16.8 | 0.847 |
| otc_token | char | neg_lam | -1.271 | -6.331 | - | 0.860 |
| otc_token | char | own_p | 0.299 | 0.017 | 17.4 | 0.850 |
| otc_token | word | p_word_w0 | 0.007 | 0.004 | 1.6 | 0.509 |
| otc_token | word | neg_lam | -62.467 | -55.851 | - | 0.435 |
| otc_token | bound | ins_w0 | 0.182 | 0.019 | 9.7 | 0.730 |
| otc_token_word | char | p_tok_w0 | 0.315 | 0.019 | 16.2 | 0.859 |
| otc_token_word | char | neg_lam | -1.430 | -5.859 | - | 0.871 |
| otc_token_word | char | own_p | 0.293 | 0.017 | 17.0 | 0.855 |
| otc_token_word | word | p_word_w0 | 0.024 | 0.007 | 3.4 | 0.517 |
| otc_token_word | word | neg_lam | -54.979 | -51.075 | - | 0.447 |
| otc_token_word | word | own_p | 0.021 | 0.006 | 3.7 | 0.516 |
| otc_token_word | bound | ins_w0 | 0.147 | 0.017 | 8.7 | 0.716 |
| otc_token_word_recip | char | p_tok_w0 | 0.346 | 0.018 | 19.2 | 0.837 |
| otc_token_word_recip | char | neg_lam | -0.915 | -6.630 | - | 0.858 |
| otc_token_word_recip | char | own_p | 0.322 | 0.015 | 21.1 | 0.839 |
| otc_token_word_recip | word | p_word_w0 | 0.013 | 0.004 | 3.3 | 0.513 |
| otc_token_word_recip | word | neg_lam | -60.897 | -55.971 | - | 0.445 |
| otc_token_word_recip | word | own_p | 0.007 | 0.002 | 3.8 | 0.505 |
| otc_token_word_recip | bound | ins_w0 | 0.242 | 0.021 | 11.4 | 0.711 |
| otc_word | char | p_tok_w0 | 0.265 | 0.017 | 15.4 | 0.804 |
| otc_word | char | neg_lam | -2.244 | -5.986 | - | 0.828 |
| otc_word | word | p_word_w0 | 0.007 | 0.003 | 2.6 | 0.509 |
| otc_word | word | neg_lam | -59.330 | -51.961 | - | 0.417 |
| otc_word | word | own_p | 0.010 | 0.002 | 4.2 | 0.510 |
| otc_word | bound | ins_w0 | 0.163 | 0.018 | 9.0 | 0.745 |
| otc_word_tuned | char | p_tok_w0 | 0.239 | 0.020 | 12.2 | 0.846 |
| otc_word_tuned | char | neg_lam | -2.205 | -5.024 | - | 0.865 |
| otc_word_tuned | word | p_word_w0 | 0.019 | 0.008 | 2.5 | 0.512 |
| otc_word_tuned | word | neg_lam | -51.555 | -47.778 | - | 0.451 |
| otc_word_tuned | word | own_p | 0.023 | 0.008 | 2.8 | 0.517 |
| otc_word_tuned | bound | ins_w0 | 0.125 | 0.011 | 10.9 | 0.733 |

**Share of units with star > 0.5** (disputed / undisputed; and how many of all starred units are disputed)

| model | char: disputed | char: undisputed | char: precision | word (p_word_w0): disputed | word: undisputed | boundary ins (w=0): disputed | boundary: undisputed |
|---|---|---|---|---|---|---|---|
| ctc | 0.231 (125) | 0.0070 (561) | 0.182 | 0.008 | 0.0026 | 0.133 | 0.0019 |
| otc_recip | 0.322 (174) | 0.0114 (916) | 0.160 | 0.009 | 0.0037 | 0.206 | 0.0033 |
| otc_token | 0.285 (154) | 0.0105 (845) | 0.154 | 0.005 | 0.0040 | 0.140 | 0.0027 |
| otc_token_word | 0.307 (166) | 0.0104 (841) | 0.165 | 0.023 | 0.0069 | 0.112 | 0.0024 |
| otc_token_word_recip | 0.333 (180) | 0.0112 (903) | 0.166 | 0.013 | 0.0037 | 0.224 | 0.0036 |
| otc_word | 0.248 (134) | 0.0081 (655) | 0.170 | 0.008 | 0.0026 | 0.121 | 0.0040 |
| otc_word_tuned | 0.217 (117) | 0.0077 (621) | 0.159 | 0.019 | 0.0072 | 0.086 | 0.0018 |

**Disputed chars with star > 0.5, by character class** (token probe, w=0)

| class | n disputed | ctc | otc_recip | otc_token | otc_token_word | otc_token_word_recip | otc_word | otc_word_tuned |
|---|---|---|---|---|---|---|---|---|
| vowel_sign | 235 | 0.24 | 0.32 | 0.30 | 0.32 | 0.33 | 0.26 | 0.19 |
| consonant | 165 | 0.22 | 0.25 | 0.25 | 0.30 | 0.30 | 0.22 | 0.26 |
| space | 49 | 0.10 | 0.37 | 0.18 | 0.27 | 0.39 | 0.20 | 0.06 |
| indep_vowel | 41 | 0.32 | 0.49 | 0.39 | 0.37 | 0.41 | 0.32 | 0.37 |
| virama | 38 | 0.29 | 0.45 | 0.39 | 0.32 | 0.42 | 0.34 | 0.26 |
| other_mark | 12 | 0.17 | 0.17 | 0.17 | 0.08 | 0.17 | 0.08 | 0.08 |

**vs CTC control** (adjusted OR of star on disputed units; ratio = OTC/CTC; dAUC with 300-sample cluster bootstrap)

| model | unit | outcome | OR CTC | OR model | ratio (model/CTC) | dAUC(p) | dAUC(-lam) |
|---|---|---|---|---|---|---|---|
| otc_recip | char | p_tok_w0>0.5 | 24.82 [18.84, 32.70] p=1.9e-115 | 28.56 [22.56, 36.15] p=8.5e-171 | 1.15 [0.94, 1.41] p=1.7e-01 | -0.005 [-0.020, +0.012] | -0.014 [-0.030, +0.001] |
| otc_recip | word | p_word_w0>0.5 | 2.28 [0.78, 6.65] p=1.3e-01 | 3.87 [1.76, 8.52] p=7.7e-04 | 1.69 [0.61, 4.71] p=3.1e-01 | +0.001 [-0.004, +0.007] | +0.010 [-0.000, +0.021] |
| otc_recip | bound | ins_w0>0.5 | 50.78 [33.56, 76.83] p=4.3e-77 | 55.36 [40.95, 74.85] p=5.5e-150 | 1.09 [0.77, 1.54] p=6.2e-01 | +0.013 [-0.015, +0.039] | - |
| otc_token | char | p_tok_w0>0.5 | 25.80 [19.69, 33.79] p=4.4e-123 | 23.62 [18.21, 30.63] p=1.3e-125 | 0.92 [0.76, 1.11] p=3.6e-01 | +0.007 [-0.008, +0.021] | -0.003 [-0.016, +0.011] |
| otc_token | word | p_word_w0>0.5 | 2.32 [0.77, 6.96] p=1.3e-01 | 2.85 [1.36, 5.98] p=5.5e-03 | 1.23 [0.48, 3.15] p=6.7e-01 | -0.000 [-0.004, +0.004] | +0.005 [-0.004, +0.015] |
| otc_token | bound | ins_w0>0.5 | 36.17 [22.94, 57.03] p=7.8e-54 | 58.63 [42.06, 81.73] p=1.8e-127 | 1.62 [1.12, 2.35] p=1.1e-02 | +0.022 [+0.001, +0.041] | - |
| otc_token_word | char | p_tok_w0>0.5 | 19.85 [14.85, 26.54] p=1.8e-90 | 31.36 [24.50, 40.15] p=9.0e-165 | 1.58 [1.28, 1.95] p=2.4e-05 | +0.019 [+0.006, +0.031] | +0.008 [-0.002, +0.020] |
| otc_token_word | word | p_word_w0>0.5 | 1.77 [0.48, 6.52] p=3.9e-01 | 2.79 [1.62, 4.81] p=2.3e-04 | 1.58 [0.44, 5.60] p=4.8e-01 | +0.008 [+0.003, +0.014] | +0.017 [+0.008, +0.027] |
| otc_token_word | bound | ins_w0>0.5 | 40.06 [25.75, 62.34] p=3.5e-60 | 52.89 [35.96, 77.78] p=2.0e-90 | 1.32 [0.94, 1.85] p=1.0e-01 | +0.008 [-0.014, +0.032] | - |
| otc_token_word_recip | char | p_tok_w0>0.5 | 25.16 [19.10, 33.15] p=3.3e-116 | 31.41 [24.92, 39.59] p=2.1e-187 | 1.25 [1.02, 1.53] p=3.4e-02 | -0.003 [-0.018, +0.014] | -0.005 [-0.018, +0.010] |
| otc_token_word_recip | word | p_word_w0>0.5 | 2.45 [0.91, 6.59] p=7.7e-02 | 5.21 [2.50, 10.86] p=1.0e-05 | 2.13 [0.89, 5.11] p=9.0e-02 | +0.004 [-0.002, +0.010] | +0.015 [+0.006, +0.026] |
| otc_token_word_recip | bound | ins_w0>0.5 | 50.11 [33.12, 75.81] p=1.2e-76 | 54.76 [40.72, 73.65] p=1.8e-154 | 1.09 [0.79, 1.51] p=5.9e-01 | +0.004 [-0.019, +0.033] | - |
| otc_word | char | p_tok_w0>0.5 | 20.49 [15.35, 27.35] p=3.3e-93 | 25.42 [19.22, 33.63] p=8.5e-114 | 1.24 [1.01, 1.52] p=3.8e-02 | -0.036 [-0.059, -0.015] | -0.035 [-0.054, -0.015] |
| otc_word | word | p_word_w0>0.5 | 2.69 [1.00, 7.20] p=5.0e-02 | 5.04 [2.15, 11.84] p=2.0e-04 | 1.88 [0.82, 4.30] p=1.4e-01 | +0.000 [-0.005, +0.005] | -0.012 [-0.022, -0.004] |
| otc_word | bound | ins_w0>0.5 | 40.19 [25.98, 62.18] p=7.9e-62 | 49.07 [34.24, 70.32] p=8.1e-100 | 1.22 [0.86, 1.73] p=2.6e-01 | +0.037 [+0.012, +0.062] | - |
| otc_word_tuned | char | p_tok_w0>0.5 | 21.28 [16.01, 28.28] p=1.6e-98 | 21.38 [15.98, 28.60] p=1.7e-94 | 1.00 [0.82, 1.24] p=9.6e-01 | +0.006 [-0.011, +0.022] | +0.002 [-0.013, +0.016] |
| otc_word_tuned | word | p_word_w0>0.5 | 1.67 [0.40, 6.94] p=4.8e-01 | 2.02 [1.09, 3.73] p=2.5e-02 | 1.21 [0.34, 4.28] p=7.7e-01 | +0.003 [-0.002, +0.009] | +0.021 [+0.012, +0.030] |
| otc_word_tuned | bound | ins_w0>0.5 | 25.58 [14.33, 45.65] p=5.4e-28 | 54.22 [37.07, 79.30] p=3.6e-94 | 2.12 [1.33, 3.39] p=1.7e-03 | +0.026 [+0.004, +0.045] | - |

## Pooled across languages (language fixed effects)

| model | unit | OR CTC | OR model | ratio (model/CTC) | langs |
|---|---|---|---|---|---|
| otc_recip | char | 17.73 [16.33, 19.24] p=0.0e+00 | 16.98 [15.84, 18.20] p=0.0e+00 | 0.96 [0.90, 1.02] p=1.6e-01 | bh bn ch hi kn mg mr mt te |
| otc_recip | word | 2.46 [1.95, 3.11] p=4.1e-14 | 5.19 [4.38, 6.15] p=9.7e-81 | 2.11 [1.73, 2.57] p=2.1e-13 | bh bn ch hi kn mg mr mt te |
| otc_token | char | 16.89 [15.56, 18.34] p=0.0e+00 | 18.05 [16.75, 19.45] p=0.0e+00 | 1.07 [1.01, 1.13] p=2.4e-02 | bh bn ch hi kn mg mr mt te |
| otc_token | word | 2.71 [2.16, 3.41] p=1.2e-17 | 4.65 [3.86, 5.60] p=4.3e-59 | 1.71 [1.42, 2.07] p=1.4e-08 | bh bn ch hi kn mg mr mt te |
| otc_token_word | char | 16.13 [14.85, 17.51] p=0.0e+00 | 17.19 [15.93, 18.55] p=0.0e+00 | 1.07 [1.00, 1.13] p=3.6e-02 | bh bn ch hi kn mg mr mt te |
| otc_token_word | word | 2.21 [1.73, 2.83] p=3.3e-10 | 2.88 [2.45, 3.39] p=4.9e-38 | 1.30 [1.07, 1.59] p=9.7e-03 | bh bn ch hi kn mg mr mt te |
| otc_token_word_recip | char | 16.95 [15.62, 18.40] p=0.0e+00 | 16.89 [15.73, 18.14] p=0.0e+00 | 1.00 [0.94, 1.06] p=9.1e-01 | bh bn ch hi kn mg mr mt te |
| otc_token_word_recip | word | 2.92 [2.36, 3.63] p=1.9e-22 | 4.01 [3.38, 4.75] p=6.3e-58 | 1.37 [1.15, 1.63] p=4.3e-04 | bh bn ch hi kn mg mr mt te |
| otc_word | char | 16.93 [15.59, 18.39] p=0.0e+00 | 17.17 [15.83, 18.63] p=0.0e+00 | 1.01 [0.95, 1.08] p=6.5e-01 | bh bn ch hi kn mg mr mt te |
| otc_word | word | 2.45 [1.94, 3.10] p=6.3e-14 | 3.21 [2.60, 3.97] p=4.2e-27 | 1.31 [1.09, 1.58] p=4.9e-03 | bh bn ch hi kn mg mr mt te |
| otc_word_tuned | char | 17.63 [16.24, 19.14] p=0.0e+00 | 16.99 [15.71, 18.39] p=0.0e+00 | 0.96 [0.91, 1.02] p=2.3e-01 | bh bn ch hi kn mg mr mt te |
| otc_word_tuned | word | 2.45 [1.93, 3.10] p=1.6e-13 | 2.30 [1.96, 2.70] p=3.4e-24 | 0.94 [0.77, 1.14] p=5.4e-01 | bh bn ch hi kn mg mr mt te |

