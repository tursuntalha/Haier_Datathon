 # DATASET DESCRİPTİON #
Data
In this competition we don’t use a single flat table, but two main data sources: (1) a time-series dataset that contains historical sales, and (2) a product master dataset that contains hierarchy and business information about each SKU. The goal is not only to predict future quantities, but also to produce consistent forecasts from SKU level up to line/category level and to correctly handle products that will phase out within the 12-month horizon. Below you can find the column descriptions and the files that will be provided.

Files
File	Description	Notes
train.csv	Historical sales data for model training. Contains monthly sales for the last 3 years for each available market × product_code combination.	Columns: market, product_code, date, quantity. Use this file to learn seasonality, trend, market-level differences and to join with product master via product_code. Missing months for a SKU/market should be treated as zero sales unless otherwise stated.
submission.csv	Example submission file that shows the exact format expected by the competition.	Typically contains: market, unique_code, date, quantity. Your final submission must follow this structure and include all rows shown in submission.csv.
Columns
1) Sales / Transaction Data
Column name	Description	Notes / Example
market	Code of the market / country / region where the sale took place. If a product is sold in multiple markets, the same product appears as separate rows.	Unmasked example: IT, ES, TR
product_code	Unique code of the sold product. This is the key column used to join with the product master data.	Unmasked example: 1111111
date	The date on which the sale occurred. This column is the time axis used to build the forecast.	e.g. 2024-05-01
quantity	The number of units sold for that market and product on that date. This is the target variable we want to forecast.	e.g. 15
2) Product Master Data
Column name	Description	Notes / Example
product_code	Unique product code. Must match product_code in the sales data one-to-one.	Join key
category	The category / product line the product belongs to. This is the level we refer to as “line” in the competition description.	Unmasked example: TV, Washing Machine, Tumble Dryer
business_line_code	Code representing the business line. Used for reporting and consolidation.	Unmasked example: FS
business_line	Descriptive name of the business line. Can be shown on dashboards or in presentations.	Unmasked example: SDA
sector	Higher-level business unit / sector to which the product belongs. Multiple business lines can be grouped under the same sector.	Unmasked example: Washing
structure_code	Industrial / production classification code of the product. Aligned with the hierarchy on the ERP / PLM side.	Unmasked example: 10
factory	The factory where the product is manufactured. For some SKUs this can change depending on sourcing or factory changes.	Unmasked example: HAIER TECH
brand	Brand of the product. Used to distinguish different brands under the Haier Europe umbrella.	Unmasked example: Haier, Candy, Hoover
start_production_date	Date when the first mass production took place. Sometimes the planned start and the actual start differ.	e.g. 2023-12-15
end_production_date	Date when production of the product is planned to stop. Sometimes the planned stop and the actual stop differ; around this period, demand is often expected to decline gradually.	e.g. 2024-12-15



# Goal / Purpose #
The goal of this datathon is to develop a robust forecasting solution that can generate 12-month demand forecasts for Haier Europe’s product portfolio, both at SKU level and at product line level.

What we expect from participants:

12-Month Horizon: All models should produce a forecast horizon of 12 months (monthly frequency, fixed horizon).

SKU-Level Forecasts: Forecasts must be generated at the most granular level, i.e. per SKU. The SKU level is critical for operational and supply planning decisions.

Line / Category-Level Consistency: When SKU-level forecasts are aggregated upwards (line / product family / category), the results should remain consistent, non-contradictory, and usable by business teams.

Phase-Out Products: Within the 12-month period there will be SKUs that are being phased out / discontinued. Models are expected not to project these SKUs as if they will keep selling forever, but to recognize stock depletion and the natural decline in sales.

For phase-out SKUs: the model should let demand converge towards zero.
For continuing SKUs: the model should preserve seasonality, trend, and promotion/campaign effects.
Deployable / Business-Ready Solution: Solutions should not only aim for leaderboard score, but should be production-ready, explainable, and robust to real-world scenarios such as missing data, sudden drops, or the appearance of new SKUs.