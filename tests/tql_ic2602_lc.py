#!/usr/bin/env/python

from tqdm import tqdm 
import chronos as cr

cluster = "IC_2602"
catalog = "CantatGaudin2018"
df_mem = cr.Cluster(cluster, catalog).query_cluster_members()

cadence = "long"
lctype = "custom"
sap_mask = "square"
outdir = f"{cluster}_{cadence[0]}c"

for gaiaid in tqdm(df_mem.source_id): 
     try: 
         fig = cr.make_tql(gaiaid=gaiaid, cadence=cadence, lctype=lctype, sap_mask=sap_mask, savetls=True, savefig=True, outdir=outdir) 
     except Exception as e: 
         print(e) 
