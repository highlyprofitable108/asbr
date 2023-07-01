UPDATE courses
        SET average_score = (
            SELECT AVG(round_score) 
            FROM rounds 
            WHERE rounds.course_num = courses.course_num
        ),
        sg_putt = (
            SELECT AVG(sg_putt)
            FROM rounds
            WHERE rounds.course_num = courses.course_num
        ),
        sg_arg = (
            SELECT AVG(sg_arg)
            FROM rounds
            WHERE rounds.course_num = courses.course_num
        ),
        sg_app = (
            SELECT AVG(sg_app)
            FROM rounds
            WHERE rounds.course_num = courses.course_num
        ),
        sg_ott = (
            SELECT AVG(sg_ott)
            FROM rounds
            WHERE rounds.course_num = courses.course_num
        ),
        sg_t2g = (
            SELECT AVG(sg_t2g)
            FROM rounds
            WHERE rounds.course_num = courses.course_num
        ),
        sg_total = (
            SELECT AVG(sg_total)
            FROM rounds
            WHERE rounds.course_num = courses.course_num
        ),
        driving_dist = (
            SELECT AVG(driving_dist)
            FROM rounds
            WHERE rounds.course_num = courses.course_num
        ),
        driving_acc = (
            SELECT AVG(driving_acc)
            FROM rounds
            WHERE rounds.course_num = courses.course_num
        ),
        gir = (
            SELECT AVG(gir)
            FROM rounds
            WHERE rounds.course_num = courses.course_num
        ),
        scrambling = (
            SELECT AVG(scrambling)
            FROM rounds
            WHERE rounds.course_num = courses.course_num
        ),
        prox_rgh = (
            SELECT AVG(prox_rgh)
            FROM rounds
            WHERE rounds.course_num = courses.course_num
        ),
        prox_fw = (
            SELECT AVG(prox_fw)
            FROM rounds
            WHERE rounds.course_num = courses.course_num
        )