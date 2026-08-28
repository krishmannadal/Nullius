"""Generate the checked-in mini corpus: ~40 documents, offline, zero downloads.

WHY THIS EXISTS. Two reasons, and the second one is the important one.

1. Tests must not need a 1.7 GB Wikipedia dump. Every unit test in this repo runs
   against this corpus in milliseconds with no network.
2. **Gold sentence indices must be derived, never typed.** Hand-counting "the claim
   is supported by sentence 3 of Marie_Curie" is precisely the off-by-one that
   corrupts an oracle condition silently. Here, gold evidence is declared as a
   *substring to locate*, and this script resolves it to a ``sent_id`` -- raising if
   the substring is absent or matches more than one sentence. A typo becomes a
   build error instead of a wrong number in a results table.

The content is hand-written and deliberately adversarial in small ways: near-duplicate
distractors ("Curie temperature", "Curie Institute"), paraphrased claims that share
little vocabulary with their gold sentence (BM25 should miss these; a dense retriever
may not), and numeric/date claims. It is a toy. It measures nothing.

Run:  python -m scripts.build_mini_corpus
Out:  data/debug/mini/corpus.jsonl, data/debug/mini/examples.jsonl
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.types import Label  # noqa: E402
from src.data.corpus import Corpus, Document  # noqa: E402
from src.data.examples import Example, label_counts, save_examples, validate_against  # noqa: E402

OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "debug" / "mini"

# --------------------------------------------------------------------------- #
# Documents. Sentences are listed explicitly -- no splitter runs here, so sent_id
# is unambiguous by construction.
# --------------------------------------------------------------------------- #

DOCUMENTS: list[tuple[str, str, list[str]]] = [
    (
        "Marie_Curie",
        "Marie Curie",
        [
            "Marie Salomea Sklodowska-Curie was a Polish and naturalised-French physicist and chemist.",
            "She was born in Warsaw, in what was then the Kingdom of Poland, part of the Russian Empire.",
            "She conducted pioneering research on radioactivity.",
            "She was the first woman to win a Nobel Prize.",
            "She was the first person to win the Nobel Prize twice, and remains the only person to win it in two different scientific fields.",
            "She shared the 1903 Nobel Prize in Physics with her husband Pierre Curie and with Henri Becquerel.",
            "She won the 1911 Nobel Prize in Chemistry alone.",
            "She died in 1934 of aplastic anaemia, believed to have been caused by prolonged exposure to radiation.",
        ],
    ),
    (
        "Pierre_Curie",
        "Pierre Curie",
        [
            "Pierre Curie was a French physicist and a pioneer in crystallography, magnetism, and radioactivity.",
            "In 1903 he received the Nobel Prize in Physics with his wife Marie Curie and Henri Becquerel.",
            "He was born in Paris on 15 May 1859.",
            "He died in a street accident in Paris in 1906.",
        ],
    ),
    (
        "Curie_temperature",
        "Curie temperature",
        [
            "In physics and materials science, the Curie temperature is the temperature above which certain materials lose their permanent magnetic properties.",
            "The Curie temperature is named after Pierre Curie, who showed that magnetism was lost at a critical temperature.",
            "For iron, the Curie temperature is approximately 770 degrees Celsius.",
        ],
    ),
    (
        "Curie_Institute_Paris",
        "Curie Institute (Paris)",
        [
            "The Curie Institute is a medical research foundation and cancer treatment centre located in Paris.",
            "It was founded in 1909 by Marie Curie and Claudius Regaud.",
            "The institute specialises in oncology research and radiotherapy.",
        ],
    ),
    (
        "Polonium",
        "Polonium",
        [
            "Polonium is a chemical element with the symbol Po and atomic number 84.",
            "It was discovered in 1898 by Marie and Pierre Curie.",
            "The element was named after Poland, the homeland of Marie Curie.",
            "Polonium is a rare and highly radioactive metal.",
        ],
    ),
    (
        "Radium",
        "Radium",
        [
            "Radium is a chemical element with the symbol Ra and atomic number 88.",
            "It was discovered by Marie and Pierre Curie in December 1898 in a uraninite sample.",
            "Radium is a silvery-white alkaline earth metal.",
            "In its pure form it is highly radioactive and luminesces a faint blue.",
        ],
    ),
    (
        "Radioactivity",
        "Radioactive decay",
        [
            "Radioactive decay is the process by which an unstable atomic nucleus loses energy by radiation.",
            "The phenomenon was discovered by Henri Becquerel in 1896.",
            "The term radioactivity was coined by Marie Curie.",
        ],
    ),
    (
        "Henri_Becquerel",
        "Henri Becquerel",
        [
            "Antoine Henri Becquerel was a French engineer and physicist.",
            "He discovered radioactivity in 1896 while working with uranium salts.",
            "He received the 1903 Nobel Prize in Physics jointly with Marie and Pierre Curie.",
        ],
    ),
    (
        "Nobel_Prize",
        "Nobel Prize",
        [
            "The Nobel Prize is a set of international awards established by the will of Alfred Nobel in 1895.",
            "The prizes were first awarded in 1901.",
            "The prizes are awarded in physics, chemistry, physiology or medicine, literature, and peace.",
            "The Nobel Memorial Prize in Economic Sciences was established separately in 1968 and is not one of the original prizes.",
        ],
    ),
    (
        "Nobel_Prize_in_Physics",
        "Nobel Prize in Physics",
        [
            "The Nobel Prize in Physics is awarded annually by the Royal Swedish Academy of Sciences.",
            "The first Nobel Prize in Physics was awarded in 1901 to Wilhelm Conrad Rontgen.",
            "As of 2023 the prize had been awarded to fewer than a dozen women.",
        ],
    ),
    (
        "Warsaw",
        "Warsaw",
        [
            "Warsaw is the capital and largest city of Poland.",
            "The city lies on the River Vistula in east-central Poland.",
            "Warsaw has a population of approximately 1.8 million residents.",
            "In the nineteenth century the city was part of the Russian Empire.",
        ],
    ),
    (
        "Paris",
        "Paris",
        [
            "Paris is the capital and most populous city of France.",
            "It is situated on the River Seine in northern France.",
            "The city is known for the Eiffel Tower, completed in 1889.",
        ],
    ),
    (
        "Poland",
        "Poland",
        [
            "Poland is a country in Central Europe.",
            "Its capital and largest city is Warsaw.",
            "Poland regained independence in 1918 after more than a century of partition.",
        ],
    ),
    (
        "University_of_Paris",
        "University of Paris",
        [
            "The University of Paris was a university in Paris, founded around 1150.",
            "It was often referred to as the Sorbonne.",
            "Marie Curie became the first woman to teach at the university, taking a professorship in 1906.",
        ],
    ),
    (
        "Irene_Joliot_Curie",
        "Irene Joliot-Curie",
        [
            "Irene Joliot-Curie was a French chemist and physicist, the daughter of Marie and Pierre Curie.",
            "She was awarded the Nobel Prize in Chemistry in 1935 with her husband Frederic Joliot-Curie.",
            "She was the second woman to win a Nobel Prize in Chemistry, after her mother.",
        ],
    ),
    (
        "Linus_Pauling",
        "Linus Pauling",
        [
            "Linus Carl Pauling was an American chemist and peace activist.",
            "He won the Nobel Prize in Chemistry in 1954 and the Nobel Peace Prize in 1962.",
            "He is one of very few people to have won more than one unshared Nobel Prize.",
        ],
    ),
    (
        "John_Bardeen",
        "John Bardeen",
        [
            "John Bardeen was an American physicist and electrical engineer.",
            "He is the only person to have won the Nobel Prize in Physics twice, in 1956 and 1972.",
        ],
    ),
    (
        "Uranium",
        "Uranium",
        [
            "Uranium is a chemical element with the symbol U and atomic number 92.",
            "It is weakly radioactive because all of its isotopes are unstable.",
            "Uranium was discovered in 1789 by Martin Heinrich Klaproth.",
        ],
    ),
    (
        "Vistula",
        "Vistula",
        [
            "The Vistula is the longest river in Poland.",
            "It has a length of 1,047 kilometres.",
            "The river flows through Krakow, Warsaw, and Torun before reaching the Baltic Sea.",
        ],
    ),
    (
        "Eiffel_Tower",
        "Eiffel Tower",
        [
            "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris.",
            "It is named after the engineer Gustave Eiffel, whose company designed and built the tower.",
            "Constructed from 1887 to 1889, it was initially criticised by some of France's leading artists.",
            "The tower is 330 metres tall.",
        ],
    ),
    # ---- distractor pages: topically adjacent, share vocabulary, settle nothing ----
    (
        "Alfred_Nobel",
        "Alfred Nobel",
        [
            "Alfred Bernhard Nobel was a Swedish chemist, engineer, and inventor.",
            "He invented dynamite in 1867.",
            "He bequeathed his fortune to establish the Nobel Prizes.",
        ],
    ),
    (
        "Royal_Swedish_Academy_of_Sciences",
        "Royal Swedish Academy of Sciences",
        [
            "The Royal Swedish Academy of Sciences is one of the Royal Academies of Sweden.",
            "It was founded in 1739.",
            "The academy selects the laureates for the Nobel Prizes in physics and chemistry.",
        ],
    ),
    (
        "Sklodowski_family",
        "Sklodowski family",
        [
            "Sklodowski is a Polish surname.",
            "Wladyslaw Sklodowski was a teacher of mathematics and physics in Warsaw.",
            "He was the father of Marie Curie.",
        ],
    ),
    (
        "Radiotherapy",
        "Radiotherapy",
        [
            "Radiotherapy is a therapy using ionising radiation to control or kill malignant cells.",
            "It is commonly applied to the cancerous tumour.",
            "Mobile radiography units were deployed during the First World War.",
        ],
    ),
    (
        "Petite_Curie",
        "Petites Curies",
        [
            "The Petites Curies were mobile radiography units used to treat wounded soldiers in the First World War.",
            "Marie Curie helped equip and drive them.",
            "The units allowed X-ray imaging close to the front lines.",
        ],
    ),
    (
        "Aplastic_anaemia",
        "Aplastic anaemia",
        [
            "Aplastic anaemia is a disease in which the body fails to produce blood cells in sufficient numbers.",
            "Exposure to ionising radiation is a known cause.",
        ],
    ),
    (
        "Chemistry",
        "Chemistry",
        [
            "Chemistry is the scientific study of the properties and behaviour of matter.",
            "It is a physical science within the natural sciences.",
        ],
    ),
    (
        "Physics",
        "Physics",
        [
            "Physics is the natural science of matter, its fundamental constituents, and its motion.",
            "Physics is one of the most fundamental scientific disciplines.",
        ],
    ),
    (
        "Krakow",
        "Krakow",
        [
            "Krakow is the second-largest city in Poland.",
            "It was the official capital of Poland until 1596.",
        ],
    ),
    (
        "Baltic_Sea",
        "Baltic Sea",
        [
            "The Baltic Sea is an arm of the Atlantic Ocean in Northern Europe.",
            "It is bounded by Denmark, Sweden, Finland, Poland, and the Baltic states.",
        ],
    ),
    (
        "Russian_Empire",
        "Russian Empire",
        [
            "The Russian Empire was an empire that spanned Eurasia from 1721 until 1917.",
            "Congress Poland was a client state of the Russian Empire established in 1815.",
        ],
    ),
    (
        "Seine",
        "Seine",
        [
            "The Seine is a river in northern France.",
            "It flows through Paris and empties into the English Channel.",
        ],
    ),
    (
        "Kingdom_of_Poland_1815",
        "Congress Poland",
        [
            "Congress Poland, formally the Kingdom of Poland, was created in 1815 by the Congress of Vienna.",
            "It was in personal union with the Russian Empire.",
        ],
    ),
    (
        "Frederic_Joliot_Curie",
        "Frederic Joliot-Curie",
        [
            "Jean Frederic Joliot-Curie was a French physicist.",
            "He shared the 1935 Nobel Prize in Chemistry with his wife Irene.",
        ],
    ),
    (
        "Crystallography",
        "Crystallography",
        [
            "Crystallography is the branch of science devoted to the study of crystals.",
            "X-ray crystallography became a major technique in the twentieth century.",
        ],
    ),
    (
        "Magnetism",
        "Magnetism",
        [
            "Magnetism is a class of physical attributes mediated by magnetic fields.",
            "Ferromagnetic materials lose their magnetisation above a critical temperature.",
        ],
    ),
    (
        "Uraninite",
        "Uraninite",
        [
            "Uraninite is a radioactive, uranium-rich mineral and ore.",
            "It was formerly known as pitchblende.",
        ],
    ),
    (
        "X_ray",
        "X-ray",
        [
            "An X-ray is a penetrating form of high-energy electromagnetic radiation.",
            "X-rays were discovered in 1895 by Wilhelm Conrad Rontgen.",
        ],
    ),
    (
        "Wilhelm_Rontgen",
        "Wilhelm Rontgen",
        [
            "Wilhelm Conrad Rontgen was a German mechanical engineer and physicist.",
            "He produced and detected electromagnetic radiation in the wavelength range known as X-rays.",
        ],
    ),
    (
        "Sorbonne",
        "Sorbonne",
        [
            "The Sorbonne is a building in the Latin Quarter of Paris.",
            "It was the historical house of the former University of Paris.",
        ],
    ),
]

# --------------------------------------------------------------------------- #
# Examples. Gold evidence is declared as (doc_id, unique substring) and RESOLVED to
# a sent_id below. A substring that is absent or ambiguous is a build error.
#
# `kind` tags the retrieval difficulty this example is meant to probe. It is a hand
# label on 12 items -- an inspection aid, not a taxonomy with any statistical weight.
# --------------------------------------------------------------------------- #

EXAMPLES: list[dict] = [
    {
        "id": "mini-001",
        "text": "Marie Curie was born in Warsaw.",
        "label": "SUPPORTS",
        "gold": [("Marie_Curie", "She was born in Warsaw")],
        "kind": "lexical-overlap",
    },
    {
        "id": "mini-002",
        "text": "Marie Curie was born in Paris.",
        "label": "REFUTES",
        "gold": [("Marie_Curie", "She was born in Warsaw")],
        "kind": "lexical-overlap",
    },
    {
        "id": "mini-003",
        "text": "Marie Curie won Nobel Prizes in two different scientific fields.",
        "label": "SUPPORTS",
        "gold": [("Marie_Curie", "the only person to win it in two different scientific fields")],
        "kind": "lexical-overlap",
    },
    {
        "id": "mini-004",
        "text": "Polonium was named after Poland.",
        "label": "SUPPORTS",
        "gold": [("Polonium", "The element was named after Poland")],
        "kind": "lexical-overlap",
    },
    {
        "id": "mini-005",
        "text": "Polonium has atomic number 88.",
        "label": "REFUTES",
        "gold": [("Polonium", "the symbol Po and atomic number 84")],
        "kind": "numeric",
    },
    {
        "id": "mini-006",
        "text": "Radium was discovered in the final month of 1898.",
        "label": "SUPPORTS",
        "gold": [("Radium", "in December 1898 in a uraninite sample")],
        "kind": "paraphrase-date",
    },
    {
        "id": "mini-007",
        "text": "The element that shares its name with a European nation was identified by a married pair of scientists.",
        "label": "SUPPORTS",
        "gold": [
            ("Polonium", "The element was named after Poland"),
            ("Polonium", "It was discovered in 1898 by Marie and Pierre Curie"),
        ],
        "kind": "paraphrase-multi-hop",
    },
    {
        "id": "mini-008",
        "text": "John Bardeen received the physics Nobel on two separate occasions.",
        "label": "SUPPORTS",
        "gold": [("John_Bardeen", "the only person to have won the Nobel Prize in Physics twice")],
        "kind": "paraphrase",
    },
    {
        "id": "mini-009",
        "text": "Pierre Curie died in Warsaw.",
        "label": "REFUTES",
        "gold": [("Pierre_Curie", "He died in a street accident in Paris in 1906")],
        "kind": "entity-confusion",
    },
    {
        "id": "mini-010",
        "text": "The Curie temperature of nickel is 358 degrees Celsius.",
        "label": "NOT ENOUGH INFO",
        "gold": [],
        "kind": "genuine-insufficiency",
    },
    {
        "id": "mini-011",
        "text": "Marie Curie's favourite colour was blue.",
        "label": "NOT ENOUGH INFO",
        "gold": [],
        "kind": "genuine-insufficiency",
    },
    {
        "id": "mini-012",
        "text": "Irene Joliot-Curie won a Nobel Prize before her mother did.",
        "label": "REFUTES",
        "gold": [
            ("Irene_Joliot_Curie", "awarded the Nobel Prize in Chemistry in 1935"),
            ("Marie_Curie", "She shared the 1903 Nobel Prize in Physics"),
        ],
        "kind": "numeric-multi-hop",
    },
    {
        "id": "mini-013",
        "text": "The Eiffel Tower is 330 metres tall and was completed in 1889.",
        "label": "SUPPORTS",
        "gold": [
            ("Eiffel_Tower", "The tower is 330 metres tall"),
            ("Eiffel_Tower", "Constructed from 1887 to 1889"),
        ],
        "kind": "conjunction",
    },
    {
        "id": "mini-014",
        "text": "Warsaw sits on the Vistula and has around 1.8 million inhabitants.",
        "label": "SUPPORTS",
        "gold": [
            ("Warsaw", "The city lies on the River Vistula"),
            ("Warsaw", "population of approximately 1.8 million"),
        ],
        "kind": "conjunction",
    },
]


def resolve_gold(corpus: Corpus, doc_id: str, needle: str) -> int:
    """(doc_id, substring) -> sent_id. Raises on absent or ambiguous matches.

    This is the whole reason gold indices in this corpus can be trusted.
    """
    doc = corpus.document(doc_id)
    hits = [i for i, s in enumerate(doc.sentences) if needle in s]
    if not hits:
        raise SystemExit(
            f"BUILD ERROR: {doc_id!r} has no sentence containing {needle!r}.\n"
            f"  sentences: {list(enumerate(doc.sentences))}"
        )
    if len(hits) > 1:
        raise SystemExit(
            f"BUILD ERROR: {needle!r} is ambiguous in {doc_id!r}; matches sentences {hits}. "
            "Lengthen the substring."
        )
    return hits[0]


def main() -> int:
    corpus = Corpus(Document(doc_id=d, title=t, sentences=tuple(s)) for d, t, s in DOCUMENTS)

    examples: list[Example] = []
    for spec in EXAMPLES:
        gold = tuple(
            (doc_id, resolve_gold(corpus, doc_id, needle)) for doc_id, needle in spec["gold"]
        )
        examples.append(
            Example(
                id=spec["id"],
                text=spec["text"],
                gold_label=Label.parse(spec["label"]),
                gold_evidence=gold,
                dataset="mini",
                meta={"kind": spec["kind"]},
            )
        )

    report = validate_against(examples, corpus)
    if not report["ok"]:
        raise SystemExit(f"BUILD ERROR: dangling gold keys: {report['dangling']}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    corpus.to_jsonl(OUT_DIR / "corpus.jsonl")
    save_examples(examples, OUT_DIR / "examples.jsonl")

    print(f"wrote {OUT_DIR}")
    for key, value in corpus.stats().items():
        print(f"  {key:<24} {value}")
    print(f"  n_examples               {len(examples)}")
    print(f"  labels                   {label_counts(examples)}")
    print(f"  gold keys resolved       {sum(len(e.gold_evidence) for e in examples)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
