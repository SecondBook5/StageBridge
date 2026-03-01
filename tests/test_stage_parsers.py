import pytest

pytest.importorskip("anndata")

from stagebridge.io.geo_snrna import parse_sample_info_from_filename
from stagebridge.io.geo_spatial import _parse_stem


def test_geo_parsers_use_lung_canonical_or_unknown():
    info = parse_sample_info_from_filename("GSM1_P1_AAH")
    assert info["stage_normalized"] == "AAH"

    info_unknown = parse_sample_info_from_filename("GSM1_P1_TURP")
    assert info_unknown["stage_normalized"] == "Unknown"

    sp = _parse_stem("GSM2_P2_MIA")
    assert sp["stage_normalized"] == "MIA"

    sp_unknown = _parse_stem("GSM2_P2_RP")
    assert sp_unknown["stage_normalized"] == "Unknown"
