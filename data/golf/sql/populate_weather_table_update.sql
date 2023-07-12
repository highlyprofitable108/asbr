UPDATE weather 
SET latitude = ?,
    longitude = ?,
    minimum_temperature = ?,
    maximum_temperature = ?,
    temperature = ?,
    precipitation_amount = ?,
    wind_gust = ?,
    wind_speed = ?,
    updated = 0
WHERE date = ?
AND course_name = ?