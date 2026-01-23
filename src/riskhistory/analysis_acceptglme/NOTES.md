# Notes about Behavioral GLME Analysis

1. The subjective value model with random slope structure (1 + gain_centered + loss_centered | sub) and (1 + gain_centered + loss_centered || sub) does not converge for Sheng2020 dataset. Hence, we set it to (1 + loss_centered | sub). 
2. Tried setting all of the choice history models random structure to be (1 + loss_centered | sub), but the following failed to converge. Hence, we have a subjective value model (with random slope per subject for loss) and the choice history model (with independent variance between loss and intercept).
    * M8 equalRange
    * M10 for equalIndifference
    * M12 for equalIndifference
    * M12 for Sheng2020
    


After model selection, we addressed the concern that the prev_choice effect 
might reflect stable individual differences in acceptance rates rather than 
true sequential dependency. Specifically, individuals who predominantly accept 
(or reject) gambles would naturally show high correlation between previous and 
current choices simply due to their baseline tendency.

To test this, we fit a model including random slopes for prev_choice:
accept ~ gain + loss + prev_choice + (1 + loss + prev_choice | sub)

This model accounts for individual differences in both baseline acceptance rates 
(random intercept) and the strength of sequential dependencies (random slope). 
The fixed effect of prev_choice remained significant (β = X, SE = Y, p = Z), 
confirming that sequential dependency is a population-level phenomenon that 
cannot be explained by individual differences in baseline acceptance tendencies alone.

Additionally, we examined whether the strength of sequential dependency varied 
with baseline acceptance rates. We extracted subject-specific random slopes for 
prev_choice and correlated them with each subject's mean acceptance rate. We 
found no significant correlation (r = X, p = Y), indicating that sequential 
dependency operates similarly across individuals regardless of their baseline 
risk attitudes.