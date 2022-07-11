These data hold the individual storyline pasture growth and storyline probaility for each of the final scenarios

{scenario}-{mod|raw}.csv
where scenario is baseline, hurt, scare
mod = farmer modification applied
raw = raw BASGRA production (no farmer modifications applied)

## keys ##
k = unique key for the storyline
ID = storyline ID (unique within the good/bad irrigation restriction suite)
log10_prob_irrigated = probability from IID for irrigated sites
log10_prob_dryland = probability from IID for dryland sites
{site}-{mode}_pg_yr1 = pasture growth for the 1 year storyline in kg DM /ha/year
{site}-{mode}_pg_m{m:02d} = pasture growth for the month m in kg DM /ha/month
irr_type = good/bad irrigation restriction suite