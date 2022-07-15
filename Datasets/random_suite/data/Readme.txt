These data hold the individual storyline pasture growth and storyline probaility for each of the final scenarios

IID_probs_pg_1y_bad_irr.hdf = where irrigation restrictions range from 50th to 99th percentile
IID_probs_pg_1y_good_irr.hdf = where irrigation restrictions range from 1st to 50th percentile

## keys ##
k = unique key for the storyline
ID = storyline ID (unique within the good/bad irrigation restriction suite)
log10_prob_irrigated = probability from IID for irrigated sites
log10_prob_dryland = probability from IID for dryland sites
{site}-{mode}_pg_yr1 = pasture growth for the 1 year storyline in kg DM /ha/year
{site}-{mode}_pg_m{m:02d} = pasture growth for the month m in kg DM /ha/month
irr_type = good/bad irrigation restriction suite