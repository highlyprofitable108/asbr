INSERT OR IGNORE INTO rounds (
                    tour, year, date, event_id, event_name, dg_id, player_name,
                    fin_text, round_num, course_num, course_name, course_par, round_score,
                    sg_putt, sg_arg, sg_app, sg_ott, sg_t2g, sg_total, driving_dist, driving_acc,
                    gir, scrambling, prox_rgh, prox_fw, new_entry
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)