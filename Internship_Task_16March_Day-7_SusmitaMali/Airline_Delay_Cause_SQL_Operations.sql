-- Creation of Database 
CREATE DATABASE airline_db;
USE airline_db;

-- Create Table
CREATE TABLE  airline_delay_cause (
    year INT,
    month INT,
    carrier CHAR(10),
    carrier_name VARCHAR(255),
    airport CHAR(10),
    airport_name VARCHAR(255),
    arr_flights FLOAT,
    arr_del15 FLOAT,
    carrier_ct FLOAT,
    weather_ct FLOAT,
    nas_ct FLOAT,
    security_ct FLOAT,
    late_aircraft_ct FLOAT,
    arr_cancelled FLOAT,
    arr_diverted FLOAT,
    arr_delay FLOAT,
    carrier_delay FLOAT,
    weather_delay FLOAT,
    nas_delay FLOAT,
    security_delay FLOAT,
    late_aircraft_delay FLOAT
);

-- Insert values in Table
INSERT INTO airline_delay_cause (year, month, carrier, carrier_name, airport, airport_name, arr_flights, arr_del15, carrier_delay, weather_delay, nas_delay, late_aircraft_delay, arr_delay)
VALUES 
(2025, 1, 'AA', 'American Airlines', 'JFK', 'New York, NY: John F. Kennedy International', 100, 20, 300, 150, 50, 400, 900),
(2025, 1, 'DL', 'Delta Air Lines', 'ATL', 'Atlanta, GA: Hartsfield-Jackson Atlanta International', 200, 30, 200, 50, 100, 300, 650),
(2025, 1, 'UA', 'United Air Lines', 'ORD', 'Chicago, IL: Chicago O\'Hare International', 150, 45, 500, 800, 200, 600, 2100);

-- Check all records from Table
SELECT * FROM airline_delay_cause;

-- Records where the carrier_name is 'Delta Air Lines'
SELECT * FROM airline_delay_cause 
WHERE carrier_name = 'Delta Air Lines';

-- Total Delay Minutes
SELECT SUM(arr_delay) AS grand_total_delay_minutes 
FROM airline_delay_cause;

-- Finding the "Root Cause" 
SELECT 
    SUM(carrier_delay) AS total_carrier, 
    SUM(weather_delay) AS total_weather, 
    SUM(late_aircraft_delay) AS total_late_aircraft
FROM airline_delay_cause;

-- the airport with the most flights (arr_flights) appears at the top.
SELECT airport_name, arr_flights 
FROM airline_delay_cause 
ORDER BY arr_flights DESC;

-- Calculating the Percentage of Flights that were Delayed
SELECT 
    carrier_name, 
    airport_name, 
    (arr_del15 / arr_flights) * 100 AS delay_percentage
FROM airline_delay_cause;

