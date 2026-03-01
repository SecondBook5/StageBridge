from stagebridge.preprocessing.stage_ontology import (
    CANONICAL_STAGE_ORDER,
    normalize_stage_label,
    ordered_transitions,
    stage_to_index,
)


def test_stage_normalization_and_order():
    assert normalize_stage_label("normal") == "Normal"
    assert normalize_stage_label("Adenocarcinoma in situ") == "AIS"
    assert normalize_stage_label("minimally invasive adenocarcinoma") == "MIA"
    assert stage_to_index("LUAD") == 4

    transitions = ordered_transitions()
    assert transitions[0] == ("Normal", "AAH")
    assert transitions[-1] == ("MIA", "LUAD")
    assert CANONICAL_STAGE_ORDER == ("Normal", "AAH", "AIS", "MIA", "LUAD")
