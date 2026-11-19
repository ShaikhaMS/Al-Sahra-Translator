"""Offline tests: the models are replaced by a stub, so nothing is downloaded.
Run with:  pytest
"""
from collections import OrderedDict

import app as webapp


class FakeTranslator:
    def __init__(self, key, langs):
        self.key, self.languages = key, langs
        self.cache = set()

    def translate(self, segments, src, tgt, beams=1):
        new = [s for s in segments if (s, src, tgt) not in self.cache]
        self.cache.update((s, src, tgt) for s in segments)
        return [f"[{tgt}] {s.upper()}" for s in segments], len(set(new))


LANGS = {"eng_Latn": "English", "arb_Arab": "Arabic (Modern Standard)", "fra_Latn": "French"}


def use_fake(key="nllb-600m"):
    m = webapp.manager
    m.current, m.loading, m.errors = key, set(), {}
    m.loaded = OrderedDict([(key, FakeTranslator(key, dict(LANGS)))])


def setup_function(_):
    use_fake()


def client():
    return webapp.app.test_client()


def test_page_has_new_name_logo_and_footer_credit_only():
    body = client().get("/").data.decode()
    assert "Al-Sahra <b>Translator</b>" in body and "TransMT" not in body and "Babel" not in body and "adu-logo.png" in body
    header = body.split("</header>")[0]
    assert "Shaikha" not in header                     # credit removed from the top
    assert "Developed by <b>Shaikha Mohammed Al Shehyari</b>" in body and "DEN702" in body
    assert "Score this translation" in body and "BLEU" in body and "chrF" in body
    assert "nllb-200-distilled" not in body


def test_model_menu_lists_four_models():
    d = client().get("/api/models").get_json()
    assert [m["key"] for m in d["models"]] == ["nllb-600m", "nllb-1.3b", "m2m-418m", "mbart-50"]


def test_status_lists_languages_for_current_model():
    s = client().get("/api/status").get_json()
    assert s["ready"] and s["model"] == "nllb-600m"
    assert {"code": "arb_Arab", "name": "Arabic (Modern Standard)", "rtl": True} in s["languages"]


def test_translate_keeps_lines_and_marks_rtl():
    r = client().post("/api/translate", json={
        "text": "Hello there. How are you?\nGood night", "model": "nllb-600m",
        "source": "eng_Latn", "target": "arb_Arab"}).get_json()
    assert r["translation"] == ("[arb_Arab] HELLO THERE. [arb_Arab] HOW ARE YOU?\n"
                                "[arb_Arab] GOOD NIGHT")
    assert r["rtl"] is True and r["sentences"] == 3


def test_only_new_sentences_are_translated():
    c = client()
    c.post("/api/translate", json={"text": "One. Two.", "source": "eng_Latn", "target": "fra_Latn"})
    r = c.post("/api/translate", json={"text": "One. Two. Three.",
                                       "source": "eng_Latn", "target": "fra_Latn"}).get_json()
    assert r["translated_now"] == 1


def test_request_for_a_model_still_loading_is_refused():
    r = client().post("/api/translate", json={"text": "Hi", "model": "mbart-50",
                                              "source": "eng_Latn", "target": "arb_Arab"})
    assert r.status_code == 503


def test_unknown_model_and_language_rejected():
    assert client().post("/api/model", json={"model": "nope"}).status_code == 400
    r = client().post("/api/translate", json={"text": "x", "source": "eng_Latn", "target": "xxx_Latn"})
    assert r.status_code == 400


def test_language_codes_map_per_family():
    assert webapp.model_code("m2m", "arb_Arab") == "ar"
    assert webapp.lang_token("m2m", "ar") == "__ar__"
    assert webapp.model_code("mbart", "arb_Arab") == "ar_AR"
    assert webapp.model_code("nllb", "arb_Arab") == "arb_Arab"


def test_switching_back_to_a_loaded_transformer_is_instant():
    m = webapp.manager
    m.loaded["m2m-418m"] = FakeTranslator("m2m-418m", dict(LANGS))
    s = client().post("/api/model", json={"model": "m2m-418m"}).get_json()
    assert s["ready"] is True and s["model"] == "m2m-418m"
    assert set(s["loaded"]) == {"nllb-600m", "m2m-418m"}


def test_least_recently_used_transformer_is_freed(monkeypatch):
    class Quick(FakeTranslator):
        def __init__(self, key):
            super().__init__(key, dict(LANGS))

        def load(self):
            pass

    monkeypatch.setattr(webapp, "Translator", Quick)
    m = webapp.ModelManager(max_loaded=2)
    for key in ["nllb-600m", "m2m-418m", "mbart-50"]:
        m.current = key
        m._load(key)
    assert list(m.loaded) == ["m2m-418m", "mbart-50"]


# ---------------------------------------------------------------------------
# BLEU and chrF (metrics.py)
# ---------------------------------------------------------------------------
import math

import metrics


def test_identical_translation_scores_100():
    s = metrics.score(["The cat sat on the mat."], ["The cat sat on the mat."])
    assert s["bleu"] == 100.0 and s["chrf"] == 100.0


def test_bleu_matches_hand_calculation():
    # precisions 5/6, 3/5, 2/4, 1/3 and no brevity penalty:
    # BLEU = (5/6 * 3/5 * 1/2 * 1/3) ** (1/4) = 53.73
    b = metrics.corpus_bleu(["the cat sat on the mat"], ["the cat sat on a mat"])
    expected = 100 * (5 / 6 * 3 / 5 * 2 / 4 * 1 / 3) ** 0.25
    assert math.isclose(b["score"], expected, rel_tol=1e-9)
    assert round(b["score"], 2) == 53.73 and b["bp"] == 1.0


def test_brevity_penalty_for_short_output():
    b = metrics.corpus_bleu(["the cat"], ["the cat sat on the mat"])
    assert math.isclose(b["bp"], math.exp(1 - 6 / 2))


def test_chrf_unigram_case_by_hand():
    # with character 1-grams only: "ab" vs "abc" -> P = 1, R = 2/3,
    # chrF(beta=2) = 5 * 1 * (2/3) / (4 * 1 + 2/3) = 71.43
    c = metrics.corpus_chrf(["ab"], ["abc"], char_order=1)
    assert round(c["score"], 2) == 71.43


def test_chrf_ignores_whitespace_and_rewards_partial_words():
    exact = metrics.corpus_chrf(["translation"], ["translation"])["score"]
    partial = metrics.corpus_chrf(["translations"], ["translation"])["score"]
    unrelated = metrics.corpus_chrf(["xyz"], ["translation"])["score"]
    assert exact == 100.0 and 80 < partial < 100 and unrelated == 0.0


def test_score_endpoint():
    r = client().post("/api/score", json={"hypothesis": "Go to the car.",
                                          "reference": "Go to the car."})
    assert r.status_code == 200 and r.get_json()["bleu"] == 100.0
    bad = client().post("/api/score", json={"hypothesis": "a\nb", "reference": "a"})
    assert bad.status_code == 400


def test_short_identical_sentence_scores_100():
    # three words have no 4-grams; BLEU must still be 100, not 0
    s = metrics.score(["مرحبا كيف أنت"], ["مرحبا كيف أنت"])
    assert s["bleu"] == 100.0 and s["chrf"] == 100.0


def test_no_common_words_scores_0():
    # Fig. 4 of the paper: Arabic output scored against an English reference
    s = metrics.score(["مرحبا كيف أنت"], ["hi there how are you"])
    assert s["bleu"] == 0.0 and s["chrf"] == 0.0
    assert s["bleu_precisions"][:3] == [16.7, 12.5, 12.5] and s["brevity_penalty"] == 0.513


def test_arabic_punctuation_is_split_for_bleu():
    assert metrics.tokenize_13a("مرحباً، كيف حالك؟") == ["مرحباً", "،", "كيف", "حالك", "؟"]
    s = metrics.score(["مرحباً، كيف حالك؟"], ["مرحباً، كيف حالك؟"])
    assert s["bleu"] == 100.0
    # a reference without the punctuation still gets a BLEU score (not 0)
    assert metrics.score(["مرحباً، كيف حالك؟"], ["مرحباً كيف حالك"])["bleu"] > 0
