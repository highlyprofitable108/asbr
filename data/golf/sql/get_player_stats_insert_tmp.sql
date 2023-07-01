INSERT INTO player_averages_temp
        SELECT
            dg_id,
            AVG(driving_dist) as driving_dist,
            AVG(driving_acc) as driving_acc,
            AVG(gir) as gir,
            AVG(scrambling) as scrambling,
            AVG(prox_rgh) as prox_rgh,
            AVG(prox_fw) as prox_fw
        FROM rounds
        GROUP BY dg_id