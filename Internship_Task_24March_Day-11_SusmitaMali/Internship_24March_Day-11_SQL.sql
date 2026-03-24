CREATE DATABASE procurement;
USE procurement;

SHOW TABLES;
-- government-procurement-via-gebiz

-- Rename table name just because - is used in previous table name
RENAME TABLE `government-procurement-via-gebiz` TO procurement_data;

-- Show all the records
SELECT * FROM procurement_data;

DESCRIBE procurement_data;

-- total of amount
SELECT SUM(awarded_amt) FROM procurement_data;

SELECT agency, SUM(awarded_amt) 
FROM procurement_data 
GROUP BY agency 
ORDER BY SUM(awarded_amt) DESC 
LIMIT 5;

-- change datefromat
UPDATE procurement_data 
SET award_date = STR_TO_DATE(award_date, '%e/%c/%Y');

SELECT YEAR(award_date) AS year, SUM(awarded_amt) AS total_spend
FROM procurement_data 
GROUP BY year
ORDER BY year;

SELECT YEAR(award_date) AS year, COUNT(*) AS wins_per_year
FROM procurement_data
WHERE supplier_name = 'NCS PTE. LTD.'
GROUP BY year
ORDER BY year;

-- -- 1. ANALYZE SPENDING TREND
SELECT 
    YEAR(award_date) AS year, 
    SUM(awarded_amt) AS total_spend
FROM procurement_data 
GROUP BY year
ORDER BY year;

-- 2. ANALYZE VENDOR PERFORMANCE TREND
SELECT 
    YEAR(award_date) AS year, 
    COUNT(*) AS contracts_won,
    SUM(awarded_amt) AS annual_revenue
FROM procurement_data
WHERE supplier_name = 'NCS PTE. LTD.'
GROUP BY year
ORDER BY year;

