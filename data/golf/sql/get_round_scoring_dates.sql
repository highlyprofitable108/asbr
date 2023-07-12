UPDATE rounds
SET date = (
    SELECT
        CASE rounds.round_num
            WHEN 4 THEN events.event_completed
            WHEN 3 THEN DATE(events.event_completed, '-1 day')
            WHEN 2 THEN DATE(events.event_completed, '-2 days')
            WHEN 1 THEN DATE(events.event_completed, '-3 days')
        END
    FROM events 
    WHERE events.event_id = rounds.event_id
    AND events.year = rounds.year
)
WHERE new_entry = 1