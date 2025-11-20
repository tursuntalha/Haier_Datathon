# Veri Seti Açıklaması

Projede kullanılan 3 farklı veri seti aşağıda gösterilmektedir.

## 1. product

Ürün bilgilerini içeren ana veri seti.

| product_code | category | business_line_code | business_line | sector | structure_code | factory | brand | start_production_date | end_production_date |
|---|---|---|---|---|---|---|---|---|---|
| PRD_7747 | CAT_01 | BLC_01 | BL_01 | SECTOR_01 | STR_01 | FACTORY_01 | BRAND_01 | 2022-05-25 | 2024-03-18 |
| PRD_12705 | CAT_02 | BLC_02 | BL_02 | SECTOR_02 | STR_02 | FACTORY_02 | BRAND_02 | 2013-12-17 | 2016-12-01 |
| PRD_13060 | CAT_03 | BLC_02 | BL_02 | SECTOR_03 | STR_03 | FACTORY_03 | BRAND_02 |  | 2020-10-05 |
| PRD_8689 | CAT_04 | BLC_02 | BL_03 | SECTOR_03 | STR_03 | FACTORY_04 | BRAND_02 | 2016-03-30 | 2018-01-26 |
| PRD_13150 | CAT_04 | BLC_02 | BL_03 | SECTOR_03 | STR_03 | FACTORY_04 | BRAND_02 | 2016-03-30 | 2018-01-26 |

## 2. sub

Tahmin sonuçlarının gönderileceği format.

| ID | unique_code | date | quantity |
|---|---|---|---|
| 0 | MKT_001-PRD_0010 | 2024-11-01 | 0 |
| 1 | MKT_001-PRD_0010 | 2024-12-01 | 0 |
| 2 | MKT_001-PRD_0010 | 2025-01-01 | 0 |
| 3 | MKT_001-PRD_0010 | 2025-02-01 | 0 |
| 4 | MKT_001-PRD_0010 | 2025-03-01 | 0 |

## 3. train

Eğitim veri seti - geçmiş satış miktarları.

| market | product_code | date | quantity |
|---|---|---|---|
| MKT_001 | PRD_0010 | 2022-01-01 | 649 |
| MKT_001 | PRD_0010 | 2022-02-01 | 1964 |
| MKT_001 | PRD_0010 | 2022-03-01 | 1505 |
| MKT_001 | PRD_0010 | 2022-04-01 | 1602 |
| MKT_001 | PRD_0010 | 2022-05-01 | 1816 |
