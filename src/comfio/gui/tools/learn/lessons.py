"""Guided-lesson catalogue for the Learn tool.

Lessons are data: adding or removing one means editing ``LESSONS``
below — the page needs no changes. All wording is comfio's own (the
CBE and StudioMatrx material inspired the idea only). Every PMV/PPD
number the page shows is computed live through ``evaluate_point``, so
``body_md`` and ``prompt`` deliberately carry no model-output values.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Step:
    label: str
    inputs: dict[str, float]  # keys ⊆ calculator Inputs fields
    prompt: str


@dataclass(frozen=True)
class Lesson:
    id: str
    title: str
    minutes: int
    body_md: str
    steps: tuple[Step, ...]
    sources: tuple[str, ...]
    links: tuple[str, ...] = ()  # tool ids


_OFFICE = {"tdb": 24.0, "tr": 24.0, "rh": 50.0, "vr": 0.1, "met": 1.2, "clo": 0.5}

LESSONS: tuple[Lesson, ...] = (
    Lesson(
        id="pmv-ppd",
        title="PMV and PPD — the heat-balance vote",
        minutes=10,
        body_md=(
            "PMV — *Predicted Mean Vote* — answers one question: if a "
            "group of people sat in this room, how would the average "
            "person rate the thermal sensation, from cold through "
            "neutral to hot? The estimate comes from the body's heat "
            "balance — heat produced by metabolism, kept by clothing, "
            "and exchanged with the room through convection (air "
            "temperature and speed), radiation (mean radiant "
            "temperature) and evaporation (humidity).\n\n"
            "PPD — *Predicted Percentage of Dissatisfied* — turns the "
            "mean vote into a head count. Even a perfectly neutral "
            "mean leaves some people unhappy, and the predicted share "
            "climbs steeply as the mean drifts. ISO 7730 and ASHRAE 55 "
            "set joint PMV/PPD acceptance limits from that curve.\n\n"
            "**What to notice:** PPD is a pure function of PMV — equal "
            "PMV always means equal PPD. And because it is a *mean*, a "
            "pass never guarantees that every occupant is comfortable."
        ),
        steps=(
            Step(
                "1 · The reference office",
                dict(_OFFICE),
                "Still air, summer clothing, seated work. Read the live "
                "PMV/PPD figure below — note how many occupants the "
                "model expects to be dissatisfied even when the mean "
                "vote is essentially neutral.",
            ),
            Step(
                "2 · Four degrees warmer",
                {**_OFFICE, "tdb": 28.0, "tr": 28.0},
                "Same room, warmer. Compare the PPD with the previous "
                "step: the dissatisfied share grows much faster than "
                "the vote itself.",
            ),
            Step(
                "3 · Compensation",
                {**_OFFICE, "tdb": 20.0, "tr": 20.0, "clo": 1.0},
                "A cool room, but occupants wear a winter ensemble. "
                "Can clothing alone pull the predicted vote back into "
                "the comfortable range?",
            ),
        ),
        sources=(
            "ISO 7730:2005",
            "ASHRAE 55-2023",
            "Tartarini & Schiavon (2020), pythermalcomfort — "
            "doi:10.1016/j.softx.2020.100578",
        ),
    ),
    Lesson(
        id="mrt",
        title="Mean radiant temperature — the half a thermometer misses",
        minutes=10,
        body_md=(
            "An air thermometer reads only the air. Your skin also "
            "trades heat by radiation with every surrounding surface — "
            "walls, ceiling, floor, windows. *Mean radiant temperature* "
            "(MRT) compresses that whole exchange into one number: the "
            "uniform surface temperature of an imaginary enclosure in "
            "which you would lose exactly the same heat by radiation.\n\n"
            "What the body experiences sits between air temperature and "
            "MRT — standards approximate it as the *operative* "
            "temperature. A cold window on a January morning or a "
            "sun-warmed ceiling can drag comfort well away from "
            "whatever the thermostat reports, and short-wave sunlight "
            "landing on the body adds a further radiant push.\n\n"
            "**What to notice:** MRT is a property of *where you sit*, "
            "not of the room as a whole. Moving closer to a cold surface "
            "raises its view factor — its share of your radiant "
            "exchange — so comfort can vary seat by seat."
        ),
        steps=(
            Step(
                "1 · Uniform room",
                dict(_OFFICE),
                "Every surface at air temperature: the predicted vote "
                "reflects the air alone.",
            ),
            Step(
                "2 · Cold glass wall",
                {**_OFFICE, "tr": 18.0},
                "Keep the air at the same temperature but drop the MRT, "
                "as if seated beside a cold glazed facade. The model "
                "reads the room as cooler than the thermostat does.",
            ),
            Step(
                "3 · Hot ceiling",
                {**_OFFICE, "tdb": 22.0, "tr": 30.0},
                "Radiant heating overhead on a cool day: the mean "
                "radiant side can overrule a perfectly reasonable air "
                "temperature.",
            ),
        ),
        sources=(
            "ISO 7726:1998",
            "ASHRAE 55-2023",
            "Arens et al. (2015), solar radiation indoors — "
            "doi:10.1016/j.buildenv.2014.09.004",
        ),
        links=("mrt",),
    ),
    Lesson(
        id="air-speed",
        title="Air speed — a cooling resource, not just a draft",
        minutes=8,
        body_md=(
            "Moving air strips heat and moisture from the skin, so "
            "raising the air speed keeps a room comfortable at a higher "
            "temperature — a slow fan can buy back a meaningful slice "
            "of cooling setpoint. The same physics cuts the other way "
            "in winter, where unwanted air movement is felt as draft.\n\n"
            "Standards put guardrails on the credit: ASHRAE 55 caps how "
            "much elevated air speed may offset warmth and demands "
            "occupant control above a modest threshold, because a "
            "breeze nobody can switch off is not a comfort feature.\n\n"
            "**What to notice:** the benefit saturates — the first "
            "gentle movement does most of the work, and gale-force fans "
            "add almost nothing more."
        ),
        steps=(
            Step(
                "1 · Warm office, still air",
                {**_OFFICE, "tdb": 27.0, "tr": 27.0},
                "A summer office just past the comfortable edge in "
                "still air.",
            ),
            Step(
                "2 · Fan on",
                {**_OFFICE, "tdb": 27.0, "tr": 27.0, "vr": 0.8},
                "Same room with a fan running. How far does the "
                "predicted vote fall — and does it cross back into the "
                "comfortable range?",
            ),
            Step(
                "3 · Winter draft",
                {
                    "tdb": 20.0, "tr": 20.0, "rh": 50.0,
                    "vr": 0.4, "met": 1.2, "clo": 1.0,
                },
                "A mild winter room with a leaky window: the same air "
                "movement now costs comfort instead of buying it.",
            ),
        ),
        sources=("ASHRAE 55-2023",),
        links=("setpoint",),
    ),
    Lesson(
        id="adaptive",
        title="Adaptive comfort — people adapt",
        minutes=10,
        body_md=(
            "Heat-balance models treat occupants as passive: same "
            "inputs, same vote. Field research — the de Dear & Brager "
            "meta-analysis is the landmark — found that people in "
            "naturally ventilated buildings do not behave that way. "
            "They open windows, change clothes, drink something cold "
            "and reset their expectations with the season, so the "
            "indoor temperatures they call comfortable track the recent "
            "outdoor weather.\n\n"
            "The adaptive models of ASHRAE 55 and EN 16798-1 encode "
            "this as a comfort line: acceptable indoor operative "
            "temperature plotted against the recent mean outdoor "
            "temperature. The line only applies where adaptation is "
            "actually possible — typically naturally ventilated spaces "
            "without mechanical cooling.\n\n"
            "**What to notice:** the heat-balance model and the "
            "adaptive line answer different questions. One predicts a "
            "vote from physics alone; the other summarises what adapted "
            "populations actually reported."
        ),
        steps=(
            Step(
                "1 · A warm spell",
                {
                    "tdb": 28.0, "tr": 28.0, "rh": 50.0, "vr": 0.3,
                    "met": 1.2, "clo": 0.5, "t_running_mean": 28.0,
                },
                "Mid-heatwave indoors without cooling. The heat-balance "
                "figure below calls this warm — 'Try it', then switch "
                "the calculator to an adaptive method (the outdoor "
                "temperature is prefilled) and see whether adapted "
                "occupants accept it.",
            ),
            Step(
                "2 · A cool spell",
                {
                    "tdb": 20.0, "tr": 20.0, "rh": 60.0, "vr": 0.15,
                    "met": 1.2, "clo": 0.7, "t_running_mean": 12.0,
                },
                "The mirror image: a cool shoulder-season day. Compare "
                "the mechanistic verdict with the adaptive line again.",
            ),
        ),
        sources=(
            "de Dear & Brager (2002), adaptive comfort — "
            "doi:10.1016/S0378-7788(02)00005-1",
            "ASHRAE 55-2023",
            "EN 16798-1:2019",
        ),
        links=("benchmark",),
    ),
    Lesson(
        id="votes-vs-models",
        title="Votes vs models — what field data say",
        minutes=8,
        body_md=(
            "The heat-balance model predicts a vote from physics; the "
            "ASHRAE Global Thermal Comfort Database II records what "
            "real occupants actually voted in real buildings. Side by "
            "side, the model follows the trend but not the crowd: the "
            "mean prediction sits offset from the mean reported vote, "
            "and the spread of real votes is far wider than any single "
            "curve.\n\n"
            "The gap has ordinary causes — guessed clothing and "
            "activity, person-to-person differences and the adaptive "
            "behaviours from the previous lesson. It is also why "
            "benchmarks matter: a prediction calibrated against a "
            "building's own votes usually outperforms one used blind.\n\n"
            "**What to notice:** disagreement is information. Plotting "
            "predicted against reported sensation across a century of "
            "field studies shows where the physics stops and the people "
            "begin."
        ),
        steps=(
            Step(
                "1 · A conditioned office",
                {
                    "tdb": 24.0, "tr": 24.0, "rh": 45.0, "vr": 0.15,
                    "met": 1.2, "clo": 0.7,
                },
                "The model's mean-vote prediction for a typical "
                "air-conditioned office. Field data show real votes "
                "scattered on both sides of this mean.",
            ),
            Step(
                "2 · Warmer, still conditioned",
                {
                    "tdb": 26.5, "tr": 26.5, "rh": 45.0, "vr": 0.15,
                    "met": 1.2, "clo": 0.6,
                },
                "Push the setpoint up: the model drifts warm, yet the "
                "database shows many occupants still voting neutral — "
                "adaptation the model cannot see.",
            ),
        ),
        sources=(
            "Földváry Ličina et al. (2018), Global Thermal Comfort "
            "Database II — doi:10.1016/j.buildenv.2018.06.022",
            "ASHRAE Global Thermal Comfort Database II, data CC0-1.0 — "
            "doi:10.6078/D1F671",
            "Graham, Parkinson & Schiavon (2021), occupant-survey "
            "lessons — doi:10.5334/bc.76",
        ),
        links=("benchmark",),
    ),
)
