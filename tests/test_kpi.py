from investor_intel.rag.kpi import extract_kpis


def test_extracts_headline_kpis(sample_text):
    kpis = {k.name: k for k in extract_kpis(sample_text)}
    assert kpis["revenue"].value == "4.2" and kpis["revenue"].unit == "billion"
    assert kpis["net_income"].value == "742" and kpis["net_income"].unit == "million"
    assert kpis["diluted_eps"].value == "5.16"
    assert kpis["operating_margin"].value == "23"
    assert kpis["yoy_revenue_growth"].value == "11"
    assert kpis["operating_cash_flow"].value == "1.1" and kpis["operating_cash_flow"].unit == "billion"
    assert kpis["total_assets"].value == "9.8" and kpis["total_assets"].unit == "billion"
    assert kpis["total_liabilities"].value == "5.4" and kpis["total_liabilities"].unit == "billion"


def test_no_kpis_in_unrelated_text():
    assert extract_kpis("The recipe calls for basil and tomatoes.") == []
