from .divisions import GENERICS, division_generics, generic_id, is_generic
from .opponents import GENERIC


def test_generic_is_the_tables_id_or_a_divisions():
    assert is_generic(GENERIC) and is_generic(generic_id("d2"))
    assert all(is_generic(g) for g in GENERICS)
    assert not is_generic("4433021") and not is_generic("generically")


def test_a_team_plays_as_its_divisions_generic_a_season_on():
    mens = division_generics("mens", 2025)
    # Lander: Division II, filed in call-it-what-you-want's 2027
    assert mens("2322") == generic_id("d2")
    # Loyola New Orleans: NAIA; Bethesda: none of them
    assert mens("100") == generic_id("naia")
    assert mens("3246") == generic_id("other")
    # a team placed in Division I -- one since moved up -- is "other"
    assert mens("2320") == generic_id("other")
    assert division_generics("womens", 2025)("2868") == generic_id("d2")
