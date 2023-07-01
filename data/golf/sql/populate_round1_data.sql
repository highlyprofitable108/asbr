CREATE TABLE round1_data AS
SELECT r.tour, r.year, datetime(r.date) AS converted_date, r.date AS original_date, r.event_id, r.event_name, r.dg_id, r.player_name,
       r.fin_text, r.round_num, r.course_num, c.course_name, c.course_par,
       r.round_score, r.sg_putt AS r_sg_putt, r.sg_arg AS r_sg_arg,
       r.sg_app AS r_sg_app, r.sg_ott AS r_sg_ott, r.sg_t2g AS r_sg_t2g,
       r.sg_total AS r_sg_total, r.driving_dist AS r_driving_dist,
       r.driving_acc AS r_driving_acc, r.gir AS r_gir, r.scrambling AS r_scrambling,
       r.prox_rgh AS r_prox_rgh, r.prox_fw AS r_prox_fw, r.new_entry,
       c.latitude, c.longitude, c.average_score AS c_average_score,
       c.sg_putt AS c_sg_putt, c.sg_arg AS c_sg_arg, c.sg_app AS c_sg_app,
       c.sg_ott AS c_sg_ott, c.sg_t2g AS c_sg_t2g, c.sg_total AS c_sg_total,
       c.driving_dist AS c_driving_dist, c.driving_acc AS c_driving_acc,
       c.gir AS c_gir, c.scrambling AS c_scrambling, c.prox_rgh AS c_prox_rgh,
       c.prox_fw AS c_prox_fw,
       w.minimum_temperature, w.maximum_temperature, w.temperature,
       w.precipitation_amount, w.wind_gust, w.wind_speed, w.updated, w.date AS weather_date
FROM rounds AS r
JOIN courses AS c ON r.course_num = c.course_num
LEFT JOIN weather AS w ON converted_date = w.date AND r.course_name = w.course_name
WHERE r.round_num = 1
  AND (c.latitude IS NOT NULL AND c.longitude IS NOT NULL)
  AND (r.round_num = 1 OR w.date IS NOT NULL);
