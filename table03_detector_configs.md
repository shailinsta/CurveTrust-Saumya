| Method | Input | Settings |
| --- | --- | --- |
| CurvTrust-PAI (full) | features + r(x) | Random Forest, 120 trees, depth 14, minimum leaf size 4, balanced subsample weights |
| RF features-only | features | Same forest settings |
| Logistic regression (LR) | features | Balanced class weights, 500 iterations |
| Gradient boosting (GB) | features | 80 trees, depth 4 |
| LR + residual | features + r(x) | As LR |
| GB + residual | features + r(x) | As GB |
| Residual conformal | r(x) only | Threshold τα, α = 0.05 |
