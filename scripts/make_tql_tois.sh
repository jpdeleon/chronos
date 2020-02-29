#import chronos as cr; import pandas as pd
#t=cr.get_tois()['TOI'].astype(int).unique(); pd.Series(t).to_csv('../data/toiids.txt', index=False)
#!/usr/bin/sh
#long cadence
cat ../data/toiids.txt | while read toi; do echo make_tql --toi=$toi -v -s -o=tois --cadence=long --sap_mask='square' --aper_radius=1; done > make_tql_tois_lc.batch
echo 'Saved: make_tql_tois_lc.batch'
#short cadence
cat ../data/toiids.txt | while read toi; do echo make_tql --toi=$toi -v -s -o=tois --cadence=short --sap_mask='pipeline'; done > make_tql_tois_sc.batch
echo 'Saved: make_tql_tois_sc.batch'
