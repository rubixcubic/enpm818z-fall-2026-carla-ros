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
