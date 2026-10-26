SELECT COUNT(*) AS applications, SUM(TARGET) AS bads, AVG(TARGET) AS bad_rate FROM modeling;
SELECT NAME_INCOME_TYPE AS segment, COUNT(*) AS applications, SUM(TARGET) AS bads,
 AVG(TARGET) AS bad_rate FROM modeling GROUP BY NAME_INCOME_TYPE ORDER BY applications DESC;
