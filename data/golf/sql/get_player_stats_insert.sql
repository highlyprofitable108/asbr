INSERT OR IGNORE INTO players (
    dg_id, player_name, sg_putt, sg_arg, sg_app, sg_ott, sg_t2g,
    sg_total, driving_dist, driving_acc, gir, scrambling, prox_rgh, prox_fw
)
SELECT 
    ?, ?, ?, ?, ?, ?, ?, ?,
    player_averages_temp.driving_dist,
    player_averages_temp.driving_acc,
    player_averages_temp.gir, 
    player_averages_temp.scrambling,
    player_averages_temp.prox_rgh,
    player_averages_temp.prox_fw
FROM player_averages_temp
WHERE player_averages_temp.dg_id = ?