CREATE TABLE model_data
AS SELECT rounds.round_score,
    rounds.date,
    weather.date,
    rounds.course_name,
    weather.course_name,
    rounds.course_num, 
    rounds.sg_putt, 
    rounds.sg_arg,
    rounds.sg_app,
    rounds.sg_ott,
    rounds.driving_dist,
    rounds.driving_acc,
    rounds.gir,
    rounds.scrambling,
    rounds.prox_rgh,
    rounds.prox_fw, 
    weather.minimum_temperature,
    weather.maximum_temperature,
    weather.temperature, 
    weather.precipitation_amount,
    weather.wind_gust,
    weather.wind_speed
FROM rounds
LEFT JOIN weather
ON DATE(rounds.date) = DATE(weather.date)
AND rounds.course_name = weather.course_name
    