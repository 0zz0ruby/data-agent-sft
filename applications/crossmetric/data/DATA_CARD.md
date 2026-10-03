# CrossMetric AI demonstration data

This project uses a semi-synthetic dataset so the prototype can demonstrate
unit economics without presenting invented operational data as real.

## Observed transaction fields

The observed transaction fields come from **UCI Online Retail II**, DOI
`10.24432/C5CG6D`, licensed under CC BY 4.0. The source contains transactions
from a UK non-store retailer between 1 December 2009 and 9 December 2011.
The demo uses a fixed-seed, uniform sample of 65,000 cleaned line items.

Observed fields include invoice, SKU, description, quantity, timestamp, unit
price, customer identifier and country. Net revenue is derived directly from
observed quantity and price.

## Simulated operational fields

The public source does not include acquisition cost, COGS, platform fees,
fulfilment cost or inventory. These fields are simulated with the fixed seed
`20261002` and explicit `sim_` prefixes. Channel assignment, advertising
clicks/impressions and inventory attributes are also simulated.

The simulation supports product prototyping and scenario analysis. It is not
evidence of a real merchant's profitability and must not be used as a financial
forecast. Recreate it with `python data/prepare_demo_data.py` after placing the
original workbook in `data/raw/online_retail_ii/online_retail_II.xlsx`.

Categories are keyword-derived, not observed retailer labels. Processed data
retains UCI attribution and CC BY 4.0 terms. Original training data is not bundled.
