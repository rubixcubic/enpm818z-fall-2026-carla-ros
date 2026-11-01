"""GP4: the traffic light message of the ground-truth helper (given).

    cd ads_pipeline && python3 -m pytest test/test_gp4_ground_truth.py -v
"""
def test_light_message_round_trip():
    from ads_pipeline.ground_truth import light_message, parse_light
    assert light_message('Red', True, 12.345) == 'Red 1 12.35'
    assert parse_light('Red 1 12.35') == ('Red', True, 12.35)
    assert parse_light(light_message('Green', False, float('nan')))[0] == 'Green'


def test_stop_line_ahead_is_positive_and_behind_is_negative():
    import math
    from ads_pipeline.ground_truth import ahead_distance
    assert ahead_distance((0.0, 0.0), 0.0, (10.0, 1.0)) == 10.0
    assert ahead_distance((0.0, 0.0), math.pi / 2, (0.0, -3.0)) == -3.0


def test_the_light_ahead_is_the_stop_line_in_the_car_lane():
    import math
    from ads_pipeline.ground_truth import light_ahead
    # The car at the origin heading +x. Stop lines: (key, (x, y), yaw).
    stops = [('mine', (20.0, 0.3), 0.0),           # 20 m ahead, our lane, our way
             ('next lane', (15.0, 3.5), 0.0),      # the lane to the left
             ('oncoming', (10.0, -0.5), math.pi),  # faces the other way
             ('far', (80.0, 0.0), 0.0)]            # beyond 50 m
    assert light_ahead((0.0, 0.0), 0.0, stops) == ('mine', 20.0)
    # Just past our line, the light still applies, so the crossing is seen.
    assert light_ahead((23.0, 0.0), 0.0, stops)[1] == -3.0
    # 6 m past it, it no longer does; the far one is 57 m ahead: none.
    assert light_ahead((26.0, 0.0), 0.0, stops) is None
