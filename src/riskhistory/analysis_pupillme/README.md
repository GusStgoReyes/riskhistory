# Pupil dilation analysis

We need to do a model per timepoint:

pupil_t ~ subjective_value + entropy + logRT + abs_gain + abs_loss +
          choice_alternation + choice


And as supplementary (i.e., is it significantly different to switch when current accept or switch when current reject)

pupil_t ~ subjective_value + entropy + logRT + abs_gain + abs_loss +
          accept_after_reject + reject_after_accept + choice

And then check if the contrast is significantly different 

library(emmeans)
emm <- emmeans(model, ~ accept_after_reject + reject_after_accept)
contrast(emm, method = "pairwise")

OR 

linearHypothesis(model, "accept_after_reject - reject_after_accept = 0")


For the random effect structure, we will try:
1. (1 + choice_alternation + subjective_value + entropy | sub)
Since we know that reward and uncertainty have been thought to influence pupil dilation. 
2. (1 + choice_alternation | sub)
Is the minimal random effect structure. 
