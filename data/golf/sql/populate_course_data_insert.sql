INSERT OR IGNORE INTO courses (
            course_num, course_name, course_par
        )
        SELECT DISTINCT
            course_num, course_name, course_par
        FROM rounds