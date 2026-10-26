# SIMULATED early-warning demonstration

Every event, event date and sales value in this directory is SIMULATED. Real applicant
IDs are sampled only to demonstrate linkage; real TARGET does not generate these outcomes.

The seeded process assigns independent artificial events with configurable probability.
Baseline daily sales are lognormal (median INR 15,000; log standard deviation 0.65),
with 15% weekly seasonality, small random organic slopes, and lognormal daily noise
(log standard deviation 0.18, mean corrected). Event cases decline linearly from
55 days before their artificial event to 35% of baseline at the event. A quarter of
non-event cases receive temporary 30-50% sales shocks lasting 20-45 days. This
deliberately generates overlap and false positives rather than perfect separation.

A fixed flag compares the most recent 30-day average with the preceding 30 days;
a decline above the configured threshold must persist for the configured days.
Only past/current sales enter a flag. Evaluation uses first alerts before artificial
events; non-event borrowers are observed over the whole artificial horizon. Precision
is event borrowers / all flagged borrowers; recall is detected / all event borrowers.
Median lead time is event day minus first alert among detected event borrowers.

The decline mechanism builds predictability into the simulation. These results cannot
establish real-world accuracy, actual default lead time, or operational value. The
threshold is fixed in config before generation; no tuning or external validation occurs.
