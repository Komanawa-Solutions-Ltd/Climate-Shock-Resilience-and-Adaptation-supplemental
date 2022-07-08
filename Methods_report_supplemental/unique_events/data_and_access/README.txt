Each of the .csv datasets here provide the daily pasture growth (kg DM/ha/day) for each of the unique events
Each row is an unique event defined by a precipitation state, temperature state, restriction percentile, and month
the columns 0-23 represents the months after the event where month 0 is the unique event.
so for instance a dry Sep month 0 = september, month 1 = october, etc.

extract_normalised_data.py will normalise the data to the reverence month
(e.g. A precip, average temperature and 50th percentile restrictions).  This can be called from the command line
and requires numpy and pandas as dependencies.