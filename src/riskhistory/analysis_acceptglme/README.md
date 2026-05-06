# Statistical Analysis Methods

## Behavioral Data Analysis

### Data Preprocessing

Behavioral data from three datasets (equalIndifference, equalRange, and Sheng2020) were combined for analysis. Gain and loss values were centered within each dataset by subtracting the midpoint of the respective ranges. This centering procedure ensures that model intercepts represent choice probabilities at the midpoint of each dataset's stimulus distribution, facilitating comparison across datasets with different ranges.

Trials were excluded from analysis based on the following criteria: (1) response times less than 0.2 seconds or greater than 3.8 seconds, (2) fewer than three valid previous trials (insufficient choice history), and (3) missing gain or loss values. Additionally, subject 56 from the NARPS dataset required response mapping correction due to an inverted button assignment, and subject 29 from the Sheng2020 dataset was excluded due to missing trial indicators in the time series data.

Choice history variables were computed within runs to prevent contamination across run boundaries. The primary choice history variable (`choice_prev1`) coded whether the previous trial's choice was accept (+1) or reject (-1). Additional lagged choice variables (`choice_prev2` through `choice_prev5`) captured choices from 2-5 trials back.

### Statistical Modeling

All behavioral models were implemented using generalized linear mixed-effects models (GLMEMs) with a binomial link function, fitted using the lme4 package (version 1.1-35.1) in R (version 4.3.1). The dependent variable was the binary choice (accept/reject) on each trial.

#### Model Specifications

**Baseline Model (M1)**: The baseline model tested the fundamental assumption of prospect theory that choices are influenced by the subjective value of gains and losses:

```
accept ~ gain_centered + loss_centered + (1 | subject) + (-1 + loss_centered | subject)
```

This model includes random intercepts and random slopes for loss sensitivity. Random slopes for gain were excluded due to convergence issues across datasets. Subject-specific parameters from this model (β₀, β_gain, β_loss) were extracted to compute trial-by-trial subjective values (SV) for subsequent models: SV = β₀ + β_gain × gain + β_loss × loss.

**Choice History Models (M2-M8)**: To test competing explanations for choice history effects, we fit a series of nested models examining whether previous choices influence current decisions independently of stimulus history or value history:

- **M2** (Primary model): M1 + choice_prev1
- **M3** (Previous value): M1 + previous subjective value
- **M4** (Value change): M1 + change in subjective value (ΔSV)
- **M5** (Previous stimuli): M1 + previous gain + previous loss
- **M6** (Value + choice): M1 + previous SV + choice_prev1
- **M7** (Value change + choice): M1 + ΔSV + choice_prev1
- **M8** (Stimuli + choice): M1 + previous gain + previous loss + choice_prev1

These models allow us to distinguish whether choice history effects reflect: (1) perseveration/switching independent of stimuli (M2), (2) carry-over of previous trial's subjective value (M3), (3) contrast effects from value changes (M4), or (4) stimulus-specific priming (M5).

**Supplementary Interaction Model (M_Sup1)**: To test whether choice history moderates sensitivity to current stimuli, we fit an interaction model:

```
accept ~ gain_centered * choice_prev1 + loss_centered * choice_prev1 + (1 | subject) + (-1 + loss_centered | subject)
```

This model tests whether gain and loss sensitivity differ following accept versus reject decisions, which would suggest that previous choices alter how decision-makers weight current options.

### Model Comparison and Inference

Model comparison was performed using Akaike Information Criterion (AIC), Bayesian Information Criterion (BIC), and Bayes Factors (BF). For the primary choice history models (M1-M8), M2 was used as the reference model to test whether alternative explanations (value history, stimulus history) provided better fits than pure choice history. For supplementary models, M2 served as the reference to test whether interactions improved fit beyond main effects.

Fixed effects were extracted using the lmerTest package, which provides Satterthwaite approximations for degrees of freedom. P-values were adjusted for multiple comparisons within each model family using the Benjamini-Hochberg false discovery rate (FDR) correction. Effects were considered statistically significant at FDR-corrected α = 0.05.

All models were fit separately for each dataset (equalIndifference and equalRange) to account for differences in stimulus ranges and allow for heterogeneity in effects across samples.

### Convergence and Model Diagnostics

All models were checked for convergence warnings. Models with convergence issues were excluded from interpretation. Variance Inflation Factors (VIFs) were computed for design matrices in complementary analyses to ensure that multicollinearity did not compromise parameter estimates.

### Software and Reproducibility

Statistical analyses were conducted using R (version 4.3.1) with the following packages: lme4 (version 1.1-35.1) for mixed-effects modeling, lmerTest (version 3.1-3) for significance tests, broom.mixed (version 0.2.9.4) for extracting model parameters, and dplyr (version 1.1.4) for data manipulation. Data preprocessing was performed in Python (version 3.12) using pandas (version 2.1.4) and numpy (version 1.26.2). All analysis code and data are available at [repository URL].
