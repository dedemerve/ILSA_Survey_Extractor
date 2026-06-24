"""
Canonical taxonomy for ILSA meta-analysis (RAG semantic depth).

Rule-based only: no API, no hallucination. Unmatched strings → Uncategorized_Contextual.
None/NaN → [IGNORE] (logged in synthesis_audit.log).
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

IGNORE_LABEL = "[IGNORE]"
UNCATEGORIZED = "Uncategorized_Contextual"
STUDENT_MOTIVATION = "Student_Motivation"
TEACHER_EFFICACY_WORKFORCE = "Teacher_Efficacy_Workforce"
THEORETICAL_META_SYNTHESIS = "Theoretical_and_Meta_Synthesis"
META_SYNTHESIS_CANONICAL_METHOD = "Theoretical_and_Meta_Synthesis"
CIVIC_ENGAGEMENT = "Civic_Engagement"
SCHOOL_EFFICIENCY_CLIMATE = "School_Efficiency_Climate"
WELLBEING_PSYCHOLOGICAL = "Wellbeing_Psychological"
ASSESSMENT_METHODOLOGY = "Assessment_Methodology"
LABOR_MARKET_OUTCOMES = "Labor_Market_Outcomes"
METADATA_FILTER_EMPIRICAL = "empirical_finding"
METADATA_FILTER_META = "theoretical_meta_synthesis"

# ILSA literature operational definitions (Appendix codebook).
OPERATIONAL_DEFINITIONS: dict[str, str] = {
    "SES": (
        "OECD/IEA composite socioeconomic background (ESCS, HOMEPOS, HISEI, PARED) "
        "capturing family wealth, parental education, and occupational status."
    ),
    "Student_Motivation": (
        "Student interest, intrinsic motivation, value beliefs, and self-concept "
        "toward learning domains in ILSA student questionnaires."
    ),
    "Math_Achievement": (
        "Mathematics proficiency on ILSA scales (e.g. PISA PV1MATH–PV10MATH, TIMSS numeracy)."
    ),
    "Reading_Achievement": (
        "Reading literacy proficiency (PISA/PIRLS reading scales and text comprehension outcomes)."
    ),
    "Science_Achievement": (
        "Science literacy/achievement (PISA/TIMSS science PVs and domain scores)."
    ),
    "ICT_Usage": (
        "Student ICT access, frequency of use, and digital learning engagement (ICTRES, ENCTUSE)."
    ),
    "Teacher_Quality_Practices": (
        "Teaching practices, instructional support, classroom climate, and teacher qualifications."
    ),
    "Wellbeing_Mental_Health": (
        "Student well-being, anxiety, stress, and life satisfaction constructs in ILSA surveys."
    ),
    "Psychometrics_Process": (
        "Psychometric scaling, latent traits (WLE/theta/EAP), item-response and process-data "
        "indicators (LPA profiles, PCCR/ACCR, measurement invariance, latent speed, log actions)."
    ),
    "Teacher_Efficacy_Workforce": (
        "TALIS/PISA teacher workforce constructs: instructional and classroom-management efficacy "
        "(ECM/ESE), job satisfaction, intention to quit, cognitive activation (COGAC), PLC, "
        "and organisational innovativeness — process drivers distinct from student achievement."
    ),
    "Academic_Resilience": (
        "Binary or latent resilience indicators (resilient vs non-resilient) in ILSA non-cognitive research."
    ),
    "Theoretical_and_Meta_Synthesis": (
        "Non-empirical synthesis rows: systematic reviews, methodology/policy discourse, "
        "heuristic migration placeholders, and literature-level outcomes without student-level "
        "effect-size targets — isolated from empirical achievement/motivation boxes for RAG."
    ),
    "Assessment_Methodology": (
        "Test design, scoring validation, DIF/Q-matrix, and automated vs human scoring agreement."
    ),
    "Labor_Market_Outcomes": (
        "PIAAC/ILSA-linked labor market outcomes: wages, employment, automation risk, "
        "overeducation penalties, task prices, and unemployment transitions."
    ),
    "Gender_Demographics": "Student sex/gender, age, and grade cohort indicators.",
    "Uncategorized_Contextual": (
        "Context-specific label that could not be aligned after primary + smart-domain rules; "
        "requires manual mapping for RAG retrieval."
    ),
}

# Smart Domain Resolver — applied only when primary patterns return Uncategorized (last resort before it).
_SMART_DOMAIN_RULES: list[tuple[str, list[str]]] = [
    (
        "Math_Achievement",
        [
            r"\bmath", r"numeracy", r"algebra", r"geometry", r"pisa\s*math",
            r"\bpm\b", r"mathematical", r"quantitative literacy",
        ],
    ),
    (
        "Reading_Achievement",
        [
            r"\bread", r"literacy", r"text comprehension", r"pisa\s*read",
            r"reading achievement", r"reading score",
        ],
    ),
    (
        "Science_Achievement",
        [
            r"\bscience", r"\bphysics", r"\bbio", r"\bchem", r"scientific literacy",
            r"pisa\s*scie",
        ],
    ),
    (
        "SES",
        [r"\bses\b", r"\bescs\b", r"\bhisei\b", r"\bpared\b", r"family wealth", r"homepos"],
    ),
    (
        TEACHER_EFFICACY_WORKFORCE,
        [
            r"\becm\b", r"\bese\b", r"efficacy in classroom",
            r"efficacy in student engagement", r"teacher self.?efficac",
            r"digital self.?efficac.*teach", r"\btalis\b", r"workforce",
            r"job satisfaction", r"intention to quit", r"tt3g50", r"jsenv", r"jspro",
            r"satisfaction with the profession", r"satisfaction with the work",
            r"work environment", r"workload", r"cognitive activation", r"\bcogac\b",
            r"professional learning communit", r"integrated professional learning",
            r"\bplc\b", r"organisational innovativeness", r"t3porgin",
            r"team innovativeness", r"\bt3team\b", r"team innovation",
            r"autonomy support", r"socioemotional support", r"encouragement and warmth",
            r"\btse\b", r"kernel causal.*\bjss\b", r"overall job satisfaction",
            r"teaching effectiveness", r"job satisf",
        ],
    ),
    (
        "Psychometrics_Process",
        [
            r"\bwle\b", r"\btheta\b", r"\beap\b", r"\birt\b", r"\bscaling\b",
            r"measurement invariance", r"item difficulty", r"error variance",
            r"\blpa\b", r"latent profile", r"latent class", r"latent speed",
            r"latent ability", r"\bpccr\b", r"\baccr\b", r"tetrachoric",
            r"classification accuracy", r"classification reliability",
            r"item response", r"item score", r"rubric-based", r"process model",
            r"latent exploration", r"engagement profile", r"inquiry performance",
            r"cheater detection", r"pvclps", r"cps score", r"colps",
            r"latent factor score", r"process profile", r"pstre",
            r"proficiency group", r"plausible values", r"bias and mse",
            r"disengagement", r"rapid guessing",
        ],
    ),
    (
        STUDENT_MOTIVATION,
        [r"motivation", r"\binterest\b", r"value belief", r"non-cognitive skills"],
    ),
    (
        "ICT_Usage",
        [r"\bict\b", r"computer", r"digital", r"technology use"],
    ),
    (
        "Teacher_Quality_Practices",
        [
            r"teaching practice", r"instructional quality", r"pedagogical",
            r"evidence-based explanation", r"asking students to provide",
        ],
    ),
    (
        "Wellbeing_Mental_Health",
        [
            r"wellbeing", r"well-being", r"anxiety", r"stress",
            r"positive affect", r"negative affect", r"\bswbp\b", r"\bswbn\b",
            r"physical health", r"life satisfaction",
        ],
    ),
    ("Metacognition_Strategies", [r"learning control", r"self-monitoring", r"\bsrl\b", r"st309"]),
    ("Occupational_Expectation", [r"future expectations", r"career expectations"]),
    (
        CIVIC_ENGAGEMENT,
        [
            r"voting", r"electoral", r"stemgedrag", r"political participation",
            r"political knowledge", r"citizen participation", r"civic engagement",
            r"\blegact\b", r"\billact\b", r"\belecpart\b", r"\bpolpart\b",
            r"protest activit", r"multicultural attitude", r"global mindedness",
            r"intercultural communication", r"cognitive mobilization",
            r"liberal to conventional", r"equal rights", r"religious establishment",
            r"federal control preference", r"\bmasque\b", r"social desirability",
            r"environmental action", r"private-sphere environmental",
            r"public-sphere environmental", r"respect for people from other cultural",
        ],
    ),
    (
        SCHOOL_EFFICIENCY_CLIMATE,
        [
            r"technical efficiency", r"\btebc\b", r"\bdea\b", r"school inefficiency",
            r"cost-efficiency", r"\bwsav\b", r"\bbsav\b", r"between-school achievement variation",
            r"within-school achievement variation", r"social segregation", r"dissimilarity index",
            r"school effectiveness", r"efficiency.?equity", r"educational poverty",
            r"classroom management", r"teaching quality", r"clear teaching",
            r"teacher collaboration", r"teacher leadership", r"teacher shortage",
            r"collaboration network", r"mutual classroom observations",
            r"collective teacher innovativeness", r"\bcti\b", r"teacher innovat",
            r"formal teacher leadership", r"informal teacher leadership",
            r"resource.?performance", r"teaching-practice centrality",
            r"exchange of teaching materials", r"positive student.?teacher relationships",
            r"collaboration in school", r"collaborative small-group learning",
            r"project-based learning", r"teaching performance network",
            r"\blmx\b", r"teacher practices \(t3tpra\)",
            r"teachers.? emphasis on teaching approaches",
            r"teachers.? teaching autonomy", r"teaching-practice centrality",
            r"teacher collaborative attitudes", r"school collective creativity",
            r"innovative teaching \(latent", r"individual innovation",
            r"teacher-supported student use of information technology",
            r"novice teacher", r"professional development needs",
            r"correlation between cognitive and non-cognitive efficiency",
            r"cross-country generalization of network structure",
            r"teacher creativity \(latent",
        ],
    ),
    (
        WELLBEING_PSYCHOLOGICAL,
        [
            r"school loneliness", r"\bloneliness\b", r"\bboredom\b", r"\benjoyment\b",
            r"\batm scale\b", r"\bgrit\b", r"growth mindset", r"emotional stability",
            r"psychosomatic", r"\beudaimonia\b", r"meaning in life", r"purpose of life",
            r"work engagement", r"emotional control", r"hostile attribution",
            r"patience", r"willingness to take risk", r"perceived value",
            r"latent emotion profile", r"preservice teachers.? emotions",
            r"children.?s creativity", r"daily scientific creativity", r"\bdsci\b",
            r"\bssci\b", r"creativity (?:flexibility|originality|total) score",
            r"creative attitude score", r"\bcas\b", r"stem competenc",
            r"empowerment-related competenc", r"adaptability-related competenc",
            r"management-related competenc", r"competitiveness \(constitutive",
            r"empathy \(constitutive", r"arrived late to school",
            r"physical activity", r"perceived societal appreciation",
            r"change in perceived societal appreciation", r"future utility beliefs",
            r"educational expectations", r"expected years of education",
            r"informal learning behav", r"non-contingent response",
            r"decision-making patterns", r"perceived competence in privacy",
            r"control strategies", r"learning difficulty \(mediator",
            r"genai responsibility", r"teachers.? importance for profession",
            r"\btip\b.*latent", r"meaning in life", r"nonformal aet",
            r"formal aet participation",             r"intention to invest in further training",
            r"teacher morality perception", r"group difference in perceived dse",
            r"learning outcomes \(mind-map", r"final research proposal mark",
        ],
    ),
    (
        ASSESSMENT_METHODOLOGY,
        [
            r"q-matrix", r"\bqrr\b", r"differential item functioning", r"\bdif\b",
            r"convergent vs discriminant validity", r"discriminant validity of domain",
            r"inter-rater reliability", r"automated scoring", r"automatic scoring",
            r"human scoring", r"hallucination rate", r"test-taking effort",
            r"test-taking engagement", r"response accuracy", r"subtask identification",
            r"cognitive diagnostic model fit", r"rasch vs skip", r"achievement level groups",
            r"mastery probabilit", r"reliability and explained common variance",
            r"bifactor model", r"overclaiming scale", r"fairness of prediction errors",
            r"disparity@", r"wasserstein distance", r"student ability vs skipping",
            r"assessment \(q\d\)", r"methods of (?:correcting|teaching)",
            r"teaching (?:planning|procedures)", r"content knowledge \(q",
            r"consistency of (?:chatgpt|latent-state)", r"profile membership",
            r"performance group \(correct", r"club membership performance",
            r"party invitation performance", r"\bu19[ab]\b",
            r"exercise response prediction", r"score prediction \(auc",
            r"detection of synthetically injected bias", r"mechanisms explaining dif",
            r"item-level differential", r"\bncdif\b",
            r"accuracy by pisa text type", r"navigation indicator",
            r"adaptive processing", r"precision \(navigation",
            r"assessment for learning", r"correct answer \(binary",
            r"personalized diagnostic report", r"multivariable analysis and prediction",
            r"control of variables", r"\bcov\) skills",
            r"dialogic teacher talk", r"task engagement.*process sequenc",
            r"country mean scores and rankings", r"pisa means \(country",
            r"pisa spans \(country",             r"dependence between pisa performance",
            r"psec indicators", r"ps-in-tre skills proficiency",
            r"student cluster membership", r"student ability vs skipping",
            r"differences in psec indicators",
        ],
    ),
    (
        WELLBEING_PSYCHOLOGICAL,
        [r"dependence between pisa performance and swb", r"copula dependence"],
    ),
    (
        "System_Policy",
        [
            r"pisa average performance", r"country-level return",
            r"student achievement mean \(state", r"interindividual inequality",
            r"output cost of (?:education|skill) mismatch",
            r"years of education \(adult", r"complete secondary school \(adult",
            r"complete bachelors \(adult", r"tertiary education attainment",
            r"household income percentile \(adult",
            r"cognitive ability mean \(state-level",
        ],
    ),
    ("ICT_Usage", [r"\bicthome\b", r"overall ai knowledge score", r"\buseutil\b"]),
    (
        LABOR_MARKET_OUTCOMES,
        [
            r"hourly wage", r"log hourly", r"log daily wage", r"hourly earnings",
            r"gross hourly wage", r"\bearnings\b", r"employment \(dummy\)",
            r"automation risk", r"job automation", r"task price", r"task intensity",
            r"task returns", r"labor productivity", r"unemployment",
            r"log-risk-ratio of exiting", r"overeducation wage", r"undereducation wage",
            r"education mismatch", r"skill mismatch", r"ai patenting",
            r"self-selection into occupations", r"co-variation of task returns",
            r"country-differential task prices", r"routine task price",
            r"manual task price", r"abstract task price",
            r"job churning occupations",
        ],
    ),
    ("Civic_Achievement", [r"civic knowledge", r"citizenship education", r"\biccs\b.*achievement"]),
    (
        "Problem_Solving_Achievement",
        [
            r"problem solving", r"creative thinking", r"\bcps\b",
            r"computational thinking composite", r"\bct performance\b",
            r"programming attainment", r"qct computational thinking",
        ],
    ),
    (
        "Reading_Achievement",
        [r"pisar achievement class", r"cil achievement profile"],
    ),
    (
        "Math_Achievement",
        [
            r"pisam achievement class", r"academic achievement \(pisa",
            r"academic performance", r"academic track recommendation",
            r"high-achievement group", r"low-achievement group",
            r"combinatorial reasoning", r"inductive reasoning \(ir\)",
            r"knowledge acquisition achievement", r"chinese achievement",
        ],
    ),
    ("Science_Achievement", [r"pisas achievement class"]),
    ("Gender_Demographics", [r"\bgender\b", r"\bsex\b", r"\bage\b"]),
    ("Parental_Involvement", [r"parent", r"family support", r"\bhme\b", r"home math"]),
    ("Prior_Achievement", [r"prior achievement", r"previous score", r"lagged"]),
    ("Academic_Resilience", [r"academic resilience", r"resilien"]),
    ("Assessment_Practices", [r"classroom assessment"]),
    ("Language_Proficiency", [r"language and communication level", r"communication level"]),
    (
        "Belonging_School_Climate",
        [r"student behavior hindering", r"hindering learning", r"disruptive behavior"],
    ),
]

_TEACHER_EFFICACY_OVERRIDE = re.compile(
    r"teacher|classroom management|talis|workforce|job satisf|ecm|ese|cogac|plc|"
    r"intention to quit|teaching information literacy|profession|work environment",
    re.IGNORECASE,
)

_SMART_COMPILED: list[tuple[str, list[re.Pattern[str]]]] = [
    (cat, [re.compile(p, re.IGNORECASE) for p in patterns])
    for cat, patterns in _SMART_DOMAIN_RULES
]

# ── ~30 ontological categories (canonical variable dictionary) ──
CANONICAL_CATEGORIES: dict[str, dict[str, Any]] = {
    "SES": {
        "description": "Socioeconomic status, home resources, parental education/occupation",
        "patterns": [
            r"\bescs\b", r"\bhomepos\b", r"\bhisei\b", r"\bpared\b", r"\bwealth\b",
            r"socio.?economic", r"ses\b", r"books at home", r"home possession",
            r"parental education", r"occupational status", r"economic.?social",
            r"family wealth", r"cultural capital",
        ],
        "seed_variations": ["ESCS", "HOMEPOS", "HISEI", "PARED", "socioeconomic status"],
    },
    "Gender_Demographics": {
        "description": "Sex, gender, age, grade level",
        "patterns": [r"\bgender\b", r"\bsex\b", r"\bage\b", r"\bgrade\b", r"boy.?girl"],
        "seed_variations": ["Gender", "Sex", "Age"],
    },
    "Immigration_Language": {
        "description": "Immigrant status, language at home, migration background",
        "patterns": [
            r"immig", r"migrant", r"language at home", r"foreign.?born",
            r"first language", r"bilingual",
        ],
        "seed_variations": ["Immigrant status", "Language at home"],
    },
    "Motivation_Interest": {
        "description": "Interest, value beliefs, intrinsic motivation",
        "patterns": [
            r"motivat", r"\binterest\b", r"value belief", r"enjoyment of",
            r"attitude toward", r"aspiration",
        ],
        "seed_variations": ["Motivation", "Interest"],
    },
    "Teacher_Efficacy_Workforce": {
        "description": (
            "TALIS/PISA teacher workforce and efficacy: ECM/ESE, job satisfaction, quit intention, "
            "COGAC, PLC, organisational innovativeness, autonomy/socioemotional support"
        ),
        "patterns": [
            r"efficacy in classroom", r"efficacy in student", r"\becm\b", r"\bese\b",
            r"teacher self.?efficac", r"instructional self.?efficac",
            r"teacher efficacy \(te\)", r"\bsdl efficacy\b",
            r"digital self.?efficac.*teach", r"self.?efficac.*teach",
            r"teacher job satisfaction", r"\btalis\b", r"workforce",
            r"job satisfaction", r"intention to quit", r"tt3g50", r"jsenv", r"jspro",
            r"satisfaction with the profession", r"satisfaction with the work",
            r"cognitive activation", r"\bcogac\b",
            r"professional learning communit", r"integrated professional learning",
            r"\bplc\b", r"organisational innovativeness", r"t3porgin",
            r"team innovativeness", r"\bt3team\b", r"team innovation",
            r"autonomy support", r"socioemotional support",
            r"overall job satisfaction", r"work environment", r"workload",
        ],
        "seed_variations": [
            "Efficacy in classroom management (ECM)",
            "Efficacy in student engagement (ESE)",
            "Intention to quit within 5 years",
        ],
    },
    "Self_Efficacy": {
        "description": "Student self-efficacy, self-concept, confidence",
        "patterns": [r"self.?efficac", r"self.?concept", r"confidence", r"belief in"],
        "seed_variations": ["Self-efficacy", "Mathematics self-concept"],
    },
    "Anxiety_Stress": {
        "description": "Test anxiety, stress, fear of failure",
        "patterns": [r"anxiety", r"stress", r"fear of failure", r"worry", r"pressure"],
        "seed_variations": ["Mathematics anxiety"],
    },
    "Belonging_School_Climate": {
        "description": "Belonging, bullying, disciplinary climate, safety",
        "patterns": [
            r"belong", r"bully", r"climate", r"safety", r"disciplinary",
            r"disruptive", r"victimization", r"school environment",
        ],
        "seed_variations": ["Sense of belonging", "Classroom climate"],
    },
    "ICT_Usage": {
        "description": "ICT access, use, digital learning, computer familiarity",
        "patterns": [
            r"\bict\b", r"computer use", r"digital", r"internet", r"technology use",
            r"online learning", r"tech.?rich",
        ],
        "seed_variations": ["ICT use", "Computer availability"],
    },
    "School_Resources": {
        "description": "School material resources, shortages, facilities",
        "patterns": [
            r"resource shortage", r"material resource", r"facility", r"equipment",
            r"staffing", r"class size", r"student.?teacher ratio",
        ],
        "seed_variations": ["School resources", "Class size"],
    },
    "Teacher_Quality_Practices": {
        "description": "Teaching practices, support, qualification, experience (non-efficacy)",
        "patterns": [
            r"instruct", r"pedagog", r"feedback", r"support from teacher",
            r"qualification", r"experience", r"teaching strateg",
        ],
        "seed_variations": ["Teacher support", "Teaching practices"],
    },
    "Parental_Involvement": {
        "description": "Parent support, involvement, homework help",
        "patterns": [
            r"parent", r"family support", r"homework help", r"home support",
            r"parental involvement", r"mother", r"father",
        ],
        "seed_variations": ["Parental involvement", "Parent support"],
    },
    "Prior_Achievement": {
        "description": "Prior scores, earlier grades, lagged achievement",
        "patterns": [
            r"prior", r"previous", r"lagged", r"earlier score", r"initial achievement",
            r"baseline achievement",
        ],
        "seed_variations": ["Prior achievement", "Previous test score"],
    },
    "Peer_Effects": {
        "description": "Peer composition, classroom aggregates",
        "patterns": [r"peer", r"class.?average", r"classmate", r"composition"],
        "seed_variations": ["Peer effects", "Classroom composition"],
    },
    "Curriculum_Instruction": {
        "description": "Instructional time, curriculum, tracking, streaming",
        "patterns": [
            r"curriculum", r"instructional time", r"tracking", r"streaming",
            r"ability group", r"lesson",
        ],
        "seed_variations": ["Instructional time", "Curriculum"],
    },
    "Math_Achievement": {
        "description": "Mathematics outcomes, PV math, numeracy",
        "patterns": [
            r"pv\d*math", r"math(?:ematics)?\s*(?:achievement|score|literacy|performance)",
            r"numeracy", r"quantitative",
            r"academic achievement \(pisa", r"academic performance",
            r"academic track recommendation", r"pisam achievement class",
            r"high-achievement group", r"low-achievement group",
            r"combinatorial reasoning", r"inductive reasoning \(ir\)",
            r"knowledge acquisition achievement", r"chinese achievement",
        ],
        "seed_variations": ["PV1MATH", "Mathematics achievement", "Math performance"],
    },
    "Science_Achievement": {
        "description": "Science outcomes and literacy",
        "patterns": [
            r"pv\d*scie", r"science\s*(?:achievement|score|literacy|performance)",
            r"scientific literacy",
        ],
        "seed_variations": ["Science achievement", "PV1SCIE"],
    },
    "Reading_Achievement": {
        "description": "Reading literacy and comprehension outcomes",
        "patterns": [
            r"pv\d*read", r"reading\s*(?:achievement|score|literacy|performance)",
            r"literacy score", r"pisar achievement class", r"cil achievement profile",
        ],
        "seed_variations": ["Reading achievement", "Reading literacy"],
    },
    "Civic_Achievement": {
        "description": "Civic knowledge, citizenship, ICCS outcomes",
        "patterns": [
            r"civic", r"citizenship", r"iccs", r"citizen.?knowledge", r"democratic",
        ],
        "seed_variations": ["Civic knowledge", "Citizenship education"],
    },
    "Problem_Solving_Achievement": {
        "description": "Problem solving, creative thinking assessments",
        "patterns": [
            r"problem.?solv", r"creative thinking", r"complex problem",
        ],
        "seed_variations": ["Problem solving", "Creative thinking"],
    },
    "Process_Data_Log": {
        "description": "Log-file, response time, process indicators",
        "patterns": [
            r"response time", r"log.?file", r"time on task",
            r"votat", r"click", r"latency",
        ],
        "seed_variations": ["Time on task", "Process data"],
    },
    "Psychometrics_Process": {
        "description": (
            "IRT/WLE/EAP scaling, LPA/latent profiles, item-level process outcomes, "
            "classification accuracy (PCCR/ACCR), measurement invariance, latent speed"
        ),
        "patterns": [
            r"\bwle\b", r"\btheta\b", r"\beap\b", r"measurement invariance",
            r"latent profile", r"\blpa\b", r"latent class", r"latent speed",
            r"\bpccr\b", r"\baccr\b", r"tetrachoric", r"classification accuracy",
            r"classification reliability", r"item response correctness",
            r"item score", r"item success", r"rubric-based",
            r"engagement profile", r"inquiry performance", r"inquiry response",
            r"cheater detection", r"latent exploration", r"latent ability",
            r"process model", r"three-factor process", r"pvclps", r"cps score",
            r"colps", r"disengagement-related indicators",
            r"latent factor score", r"process profile label", r"\bpstre\b",
            r"proficiency group mean", r"plausible values\)", r"bias and mse",
            r"process-incorporated", r"item3 response", r"human vs ann",
            r"\bscaling\b", r"item difficulty", r"measurement error",
            r"disengagement", r"rapid guessing", r"latent disengagement",
        ],
        "seed_variations": [
            "WLE theta",
            "LPA profiles",
            "PCCR",
            "Measurement invariance",
        ],
    },
    "Wellbeing_Mental_Health": {
        "description": "Well-being, life satisfaction, mental health",
        "patterns": [
            r"well.?being", r"wellbeing", r"life satisfaction", r"mental health",
            r"happiness", r"flourish",
        ],
        "seed_variations": ["Well-being", "Life satisfaction"],
    },
    "System_Policy": {
        "description": "Country/system policy, GDP, tracking age, expenditure",
        "patterns": [
            r"\bgdp\b", r"gini", r"expenditure", r"policy", r"tracking age",
            r"system level", r"country level",
        ],
        "seed_variations": ["GDP per capita", "Education expenditure"],
    },
    "School_Type_Composition": {
        "description": "Public/private, urban/rural, school type",
        "patterns": [
            r"private school", r"public school", r"urban", r"rural", r"school type",
            r"denominational", r"selective",
        ],
        "seed_variations": ["School type", "Urban/rural location"],
    },
    "Language_Proficiency": {
        "description": "Language proficiency, reading fluency as predictor",
        "patterns": [r"language proficiency", r"fluency", r"vocabulary"],
        "seed_variations": ["Language proficiency"],
    },
    "Occupational_Expectation": {
        "description": "Career expectations, occupational plans",
        "patterns": [r"occupational", r"career expect", r"job expect", r"ambition"],
        "seed_variations": ["Occupational expectations"],
    },
    "Assessment_Practices": {
        "description": "Testing practices, assessment policy (non-PV outcome)",
        "patterns": [r"assessment practice", r"high.?stakes test", r"accountability"],
        "seed_variations": ["Assessment practices"],
    },
    "Dropout_Retention": {
        "description": "Dropout risk, retention, early school leaving",
        "patterns": [r"dropout", r"early school leaving", r"retention", r"truancy"],
        "seed_variations": ["Dropout risk"],
    },
    "Homework_Study_Time": {
        "description": "Homework, study time, learning time",
        "patterns": [
            r"homework", r"study time", r"learning time", r"out.?of.?school study",
        ],
        "seed_variations": ["Homework time", "Study time"],
    },
    "Metacognition_Strategies": {
        "description": "Learning strategies, metacognition, self-regulation",
        "patterns": [
            r"metacogn", r"learning strateg", r"self.?regulat", r"effort regulat",
        ],
        "seed_variations": ["Learning strategies", "Metacognition"],
    },
    "Academic_Resilience": {
        "description": "Academic resilience and non-cognitive protective factors",
        "patterns": [
            r"academic resilience", r"resilien", r"non-resilient",
        ],
        "seed_variations": ["Academic resilience"],
    },
    "Civic_Engagement": {
        "description": "Political/civic participation, voting, multicultural attitudes",
        "patterns": [
            r"voting intention", r"expected electoral", r"expected participation",
            r"political participation", r"political knowledge", r"citizen participation",
            r"\blegact\b", r"\billact\b", r"\belecpart\b", r"\bpolpart\b",
            r"stemgedrag", r"multicultural attitude", r"\bmasque\b",
            r"global mindedness", r"intercultural communication",
            r"cognitive mobilization", r"liberal to conventional",
            r"support for equal rights", r"federal control preference",
            r"environmental action", r"information management \(infom\)",
            r"information evaluation \(infoe\)",
        ],
        "seed_variations": [
            "Expected electoral participation (ELECPART)",
            "Multicultural attitudes (MASQUE factors: Know, Care, Act)",
        ],
    },
    "School_Efficiency_Climate": {
        "description": "School efficiency, inequality, leadership, collaboration climate",
        "patterns": [
            r"technical efficiency", r"\btebc\b", r"school inefficiency",
            r"\bwsav\b", r"\bbsav\b", r"social segregation", r"dissimilarity index",
            r"school effectiveness", r"efficiency.?equity", r"educational poverty",
            r"classroom management", r"teaching quality", r"clear teaching",
            r"teacher collaboration", r"teacher leadership", r"teacher shortage",
            r"collaboration network", r"collaborative knowledge construction",
            r"collective teacher innovativeness",
            r"formal teacher leadership", r"informal teacher leadership",
            r"within-school inequality", r"heterogeneous effect difference: oecd",
            r"collaborative knowledge construction",
        ],
        "seed_variations": [
            "School technical efficiency (bias-corrected TEBC)",
            "Social segregation across schools (Dissimilarity Index D)",
        ],
    },
    "Wellbeing_Psychological": {
        "description": "Psychological traits and socio-emotional outcomes (non-clinical)",
        "patterns": [
            r"school loneliness", r"\bloneliness\b", r"\bboredom\b", r"\benjoyment\b",
            r"\batm scale\b", r"\bgrit\b", r"growth mindset", r"emotional stability",
            r"psychosomatic", r"\beudaimonia\b", r"meaning in life", r"purpose of life",
            r"work engagement", r"emotional control", r"hostile attribution",
            r"patience", r"risk-taking", r"perceived value",
            r"children.?s creativity", r"daily scientific creativity", r"\bdsci\b",
            r"\bssci\b", r"creative attitude score", r"stem competenc",
            r"educational expectations", r"future utility beliefs",
            r"competitiveness \(constitutive", r"empathy \(constitutive",
            r"arrived late to school", r"physical activity",
            r"perceived societal appreciation", r"preservice teachers.? emotions",
        ],
        "seed_variations": [
            "School loneliness (weighted mean score)",
            "Enjoyment (ATM scale)",
        ],
    },
    "Labor_Market_Outcomes": {
        "description": "Wages, employment, automation risk, overeducation, unemployment (PIAAC/ILSA)",
        "patterns": [
            r"hourly wage", r"log hourly", r"log daily wage", r"hourly earnings",
            r"\bearnings\b", r"employment \(dummy\)", r"automation risk",
            r"job automation", r"task price", r"task intensity", r"task returns",
            r"labor productivity", r"unemployment", r"log-risk-ratio of exiting",
            r"overeducation wage", r"undereducation wage", r"education mismatch",
            r"skill mismatch", r"ai patenting", r"self-selection into occupations",
            r"co-variation of task returns", r"country-differential task prices",
            r"routine task price", r"manual task price", r"abstract task price",
            r"mediated share of overeducation",
        ],
        "seed_variations": [
            "Log hourly wages",
            "Automation risk (individual-level, 0–1)",
            "Overeducation wage penalty (vs appropriately matched)",
        ],
    },
    "Assessment_Methodology": {
        "description": "Assessment design, scoring validation, DIF, automated scoring",
        "patterns": [
            r"q-matrix", r"\bqrr\b", r"differential item functioning", r"\bdif\b",
            r"convergent vs discriminant validity", r"inter-rater reliability",
            r"automated scoring", r"automatic scoring", r"hallucination rate",
            r"test-taking effort", r"test-taking engagement", r"response accuracy",
            r"cognitive diagnostic model fit", r"achievement level groups",
            r"mastery probabilit", r"reliability and explained common variance",
            r"bifactor model", r"overclaiming scale", r"student ability vs skipping",
            r"assessment \(q\d\)", r"methods of teaching", r"teaching procedures",
            r"content knowledge \(q", r"profile membership", r"performance group",
            r"exercise response prediction", r"item-level differential", r"\bncdif\b",
            r"mechanisms explaining dif", r"fairness of prediction errors",
            r"t-sec overall score", r"\bmatsec\b", r"individual marking criteria",
            r"score difference \(chatgpt", r"failure vs \(partial\) success on",
            r"knowledge-state learning trajectories",
            r"multivariable causal reasoning",
            r"task performance:.*tpack",
            r"performance gap under low vs high stakes",
        ],
        "seed_variations": [
            "Q-matrix recovery rate (QRR)",
            "Automated scoring agreement (accuracy) vs human scoring",
        ],
    },
    "Theoretical_and_Meta_Synthesis": {
        "description": (
            "Literature synthesis, policy/methodology discourse, heuristic migration labels, "
            "and framework-report narrative outcomes (not student-level empirical targets)"
        ),
        "patterns": [
            r"heuristic migration",
            r"literature synthesis",
            r"not student-level prediction",
            r"no single student-level analytic",
            r"assessment context not specified",
            r"primary study outcome \(heuristic",
            r"primary analytic outcome \(inferred",
            r"ILSA literature synthesis",
            r"systematic review",
            r"predicted with measurable ML outcomes",
            r"predicts societal cognitive",
            r"derived binary variables",
            r"discernibility of authorship",
            r"correspondence between screened countries",
            r"intelligence vs personality",
            r"accuracy by question format",
            r"accuracy by pisa difficulty",
            r"accuracy by bloom",
            r"cdm model fit quality",
            r"automated scoring correctness",
            r"how different ilsa",
        ],
        "seed_variations": [
            "Primary study outcome (heuristic migration)",
            "Literature synthesis outcome (not student-level prediction)",
            "ILSA literature synthesis (systematic review; no single student-level analytic micro-dataset)",
        ],
    },
    "Uncategorized_Contextual": {
        "description": "Context-specific labels that do not match canonical ontology",
        "patterns": [],
        "seed_variations": [],
    },
}

# Compile patterns once
_COMPILED: dict[str, list[re.Pattern[str]]] = {
    cat: [re.compile(p, re.IGNORECASE) for p in meta.get("patterns", [])]
    for cat, meta in CANONICAL_CATEGORIES.items()
    if cat != UNCATEGORIZED
}

# ── Methodological taxonomy (mandatory academic grouping) ──
METHOD_CANONICAL_MAP: dict[str, list[str]] = {
    "Traditional_Stats": [
        "regression", "anova", "sem", "structural equation", "hlm", "hierarchical linear",
        "multilevel", "cfa", "confirmatory factor", "correlation", "ols", "logit",
        "probit", "path analysis", "latent variable", "mplus", "lavaan", "rubin",
        "plausible value", "irt", "wle", "delta method", "dea", "sfa", "efficiency",
        "fixed effect", "random effect", "difference in difference", "propensity",
    ],
    "Ensemble_Learning": [
        "random forest", "xgboost", "gradient boost", "lightgbm", "catboost",
        "adaboost", "extra trees", "bagging", "stacking ensemble", "ensemble",
    ],
    "Deep_Learning": [
        "neural network", "deep learning", "rnn", "lstm", "cnn", "transformer",
        "mlp", "deep neural", "autoencoder",
    ],
    "Supervised_General": [
        "svm", "support vector", "knn", "k-nearest", "decision tree", "logistic",
        "naive bayes", "linear discriminant", "classification tree", "annfis", "anfis",
    ],
    "Unsupervised": [
        "k-means", "kmeans", "cluster", "pca", "principal component", "factor analysis",
        "latent class", "latent profile", "dbscan", "topic model",
    ],
}

_SENTINEL_MARKERS = (
    "n/a:", "not reported", "missing", "uncategorized", "[ignore]",
)

_META_SYNTHESIS_EXACT: frozenset[str] = frozenset({
    "Primary study outcome (heuristic migration)",
    "Literature synthesis outcome (not student-level prediction)",
    "Primary analytic outcome (inferred from extraction)",
    "ILSA assessment context not specified in extraction (heuristic migration)",
    "ILSA literature synthesis (systematic review; no single student-level analytic micro-dataset)",
})

_META_SYNTHESIS_SUBSTRINGS: tuple[str, ...] = (
    "heuristic migration",
    "literature synthesis",
    "not student-level prediction",
    "no single student-level analytic",
    "assessment context not specified",
)

_NON_EMPIRICAL_STUDY_FILTERS: frozenset[str] = frozenset({
    "Technical/Assessment Framework",
    "Descriptive National Report",
})

_NON_EMPIRICAL_SOURCE_CATEGORIES: frozenset[str] = frozenset({
    "review_article",
    "methodology_paper",
    "technical_report",
    "book_chapter",
})

_META_SYNTHESIS_PROSE_RE: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"^how (?:these|national|the |students|CIL|to use|Chinese|PISA|TIMSS|PIRLS)",
        r"^that (?:achievement|international|student achievement|PISA|PIRLS|TIMSS)",
        r"data representation",
        r"guide item development",
        r"constrain what statistical comparisons",
        r"components relate to foundational print",
        r"^\d+% of between-school variance",
        r"content-domain expansion",
        r"attitudes toward authoritarianism achievement",
        r"pupil, home achievement",
        r"egalitarian attitudes achievement",
        r"^knowledge\s*$",
        r"^scale achievement",
        r"engagement measures achievement",
        r"\)\s*achievement\s*$",
        r"^how students['\u2019]? social media use",
        r"regional student questionnaire module administered",
        r"^how CIL and CT constructs are structured",
        r"sampling weights \(TOTWGT",
        r"were international achievement surveys, and it co",
    )
)


def _norm(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()


def is_ignore_value(value: Any) -> bool:
    text = _norm(value).lower()
    if not text:
        return True
    return any(text.startswith(m) for m in _SENTINEL_MARKERS)


def is_theoretical_meta_synthesis_label(
    raw: Any,
    *,
    study_filter: str = "",
    source_category: str = "",
    document_class: str = "",
) -> bool:
    """
    Detect non-empirical / literature-synthesis target labels (heuristic migration, reviews).

    Used after primary + smart mapping fails, and for non-empirical study_filter rows.
    """
    text = _norm(raw)
    if not text:
        return False
    if text in _META_SYNTHESIS_EXACT:
        return True
    lowered = text.lower()
    if any(s in lowered for s in _META_SYNTHESIS_SUBSTRINGS):
        return True
    if any(pat.search(text) for pat in _META_SYNTHESIS_PROSE_RE):
        return True

    sf = _norm(study_filter)
    sc = _norm(source_category).lower()
    dc = _norm(document_class).lower()
    if sf in _NON_EMPIRICAL_STUDY_FILTERS:
        return True
    if sc in _NON_EMPIRICAL_SOURCE_CATEGORIES and (
        len(text) > 35 or lowered.startswith(("how ", "that ", "literature "))
    ):
        return True
    if dc == "technical_report" and (
        "heuristic" in lowered or "literature" in lowered or len(text) > 50
    ):
        return True
    return False


def metadata_filter_flag_for(canonical_variable: str) -> str:
    """RAG routing flag: empirical effect retrieval vs meta-synthesis layer."""
    if canonical_variable == THEORETICAL_META_SYNTHESIS:
        return METADATA_FILTER_META
    if canonical_variable in (IGNORE_LABEL, UNCATEGORIZED):
        return "excluded"
    return METADATA_FILTER_EMPIRICAL


def build_rag_navigation_map() -> dict[str, Any]:
    """Navigation map for RAG query routing (empirical vs theoretical/meta layer)."""
    return {
        "version": "1.0",
        "description": (
            "Route user questions to empirical canonical variables or to "
            "Theoretical_and_Meta_Synthesis for literature, policy, and methodology discourse."
        ),
        "routes": [
            {
                "metadata_filter_flag": METADATA_FILTER_EMPIRICAL,
                "canonical_variable_families": [
                    "Math_Achievement",
                    "Reading_Achievement",
                    "Science_Achievement",
                    "SES",
                    "Student_Motivation",
                    "Teacher_Efficacy_Workforce",
                    "Psychometrics_Process",
                    "Civic_Engagement",
                    "School_Efficiency_Climate",
                    "Wellbeing_Psychological",
                    "Assessment_Methodology",
                    "Labor_Market_Outcomes",
                ],
                "example_queries": [
                    "What is the effect of SES on mathematics achievement?",
                    "How strong is the association between ICT use and science scores?",
                    "Does teacher efficacy predict student engagement?",
                ],
            },
            {
                "metadata_filter_flag": METADATA_FILTER_META,
                "canonical_variable": THEORETICAL_META_SYNTHESIS,
                "canonical_method": META_SYNTHESIS_CANONICAL_METHOD,
                "example_queries": [
                    "What does the literature say about ILSA policy debates?",
                    "How has research on PISA methodology evolved?",
                    "Summarize systematic reviews on large-scale assessment frameworks.",
                    "Historical development of international education surveys.",
                ],
            },
        ],
        "routing_rule": (
            "If the query asks for effect sizes, predictors, or student-level impacts, "
            f"use {METADATA_FILTER_EMPIRICAL}. If it asks about literature trends, policy "
            f"arguments, methodology papers, or assessment-framework discourse, use "
            f"{METADATA_FILTER_META}."
        ),
    }


def smart_domain_resolver(text: str) -> str | None:
    """
    Second-pass resolver for noisy LLM target strings (Smart Domain Resolver).
    Returns None if no rule matches — caller may then use Uncategorized_Contextual.
    """
    lowered = text.lower()
    for category, patterns in _SMART_COMPILED:
        for pat in patterns:
            if pat.search(lowered):
                return category
    return None


def map_variable_to_canonical(
    raw: Any,
    *,
    use_smart_resolver: bool = True,
    study_filter: str = "",
    source_category: str = "",
    document_class: str = "",
) -> str:
    """
    Map variable_name or target_variable to one canonical category.

    Order: primary regex → smart domain resolver → Theoretical_and_Meta_Synthesis
    (meta labels / non-empirical context) → Uncategorized_Contextual (last resort).
    """
    if is_ignore_value(raw):
        return IGNORE_LABEL
    text = _norm(raw)
    if not text:
        return IGNORE_LABEL

    for category, patterns in _COMPILED.items():
        for pat in patterns:
            if pat.search(text):
                if category == "Motivation_Interest":
                    return STUDENT_MOTIVATION
                if category == "Self_Efficacy" and _TEACHER_EFFICACY_OVERRIDE.search(text):
                    return TEACHER_EFFICACY_WORKFORCE
                return category

    if use_smart_resolver:
        resolved = smart_domain_resolver(text)
        if resolved:
            return resolved

    if is_theoretical_meta_synthesis_label(
        text,
        study_filter=study_filter,
        source_category=source_category,
        document_class=document_class,
    ):
        return THEORETICAL_META_SYNTHESIS

    return UNCATEGORIZED


def map_method_to_canonical(ml_text: Any) -> str:
    """Map ml_techniques / ml_primary to mandatory method family."""
    if is_ignore_value(ml_text):
        return IGNORE_LABEL
    text = _norm(ml_text).lower()
    if not text:
        return IGNORE_LABEL

    # Order: more specific ensembles/deep before broad traditional
    priority = (
        "Deep_Learning",
        "Ensemble_Learning",
        "Unsupervised",
        "Supervised_General",
        "Traditional_Stats",
    )
    hits: list[str] = []
    for family, keywords in METHOD_CANONICAL_MAP.items():
        if any(kw in text for kw in keywords):
            hits.append(family)
    if not hits:
        return UNCATEGORIZED
    for fam in priority:
        if fam in hits:
            return fam
    return hits[0]


def harmonize_effect_trend(value: Any) -> str:
    """
    Effect-Trend Harmonization: only Positive | Negative | Null.

    Null = non-significant, not reported, or ambiguous conclusions.
    """
    if is_ignore_value(value):
        return "Null"
    text = _norm(value)
    if text in ("Positive", "Negative", "Null"):
        return text
    return infer_effect_trend(text)


def infer_effect_trend(conclusion: Any) -> str:
    """Extract effect direction from conclusion text → Positive | Negative | Null."""
    text = _norm(conclusion).lower()
    if not text:
        return "Null"

    null_pats = [
        r"no (?:significant |clear )?(?:effect|difference|impact|association)",
        r"not significant", r"non.?significant", r"null effect",
        r"similar (?:levels|performance)", r"no evidence",
        r"did not (?:find|show|reveal)", r"not reported", r"insignificant",
    ]
    pos_pats = [
        r"positively", r"positive (?:effect|association|relationship|impact)",
        r"increase[ds]?", r"higher", r"improve[ds]?", r"stronger", r"benefit",
        r"predicted (?:higher|better)", r"associated with (?:higher|better)",
    ]
    neg_pats = [
        r"negatively", r"negative (?:effect|association|relationship|impact)",
        r"decrease[ds]?", r"lower", r"reduce[ds]?", r"worse", r"weaker",
        r"predicted (?:lower|worse)", r"associated with (?:lower|worse)",
    ]

    null_score = sum(1 for p in null_pats if re.search(p, text))
    pos_score = sum(1 for p in pos_pats if re.search(p, text))
    neg_score = sum(1 for p in neg_pats if re.search(p, text))

    if pos_score > neg_score and pos_score > 0:
        return "Positive"
    if neg_score > pos_score and neg_score > 0:
        return "Negative"
    return "Null"


def build_taxonomy_maps(
    variable_names: Iterable[str],
) -> tuple[dict[str, Any], dict[str, str]]:
    """
    Build taxonomy_map.json structure and flat variable_taxonomy_map.
    Populates variations list per category from matched raw strings.
    """
    categories: dict[str, dict[str, Any]] = {}
    for cat, meta in CANONICAL_CATEGORIES.items():
        categories[cat] = {
            "description": meta["description"],
            "variations": list(meta.get("seed_variations", [])),
        }

    flat_map: dict[str, str] = {}
    for raw in variable_names:
        if is_ignore_value(raw):
            continue
        key = _norm(raw)
        if not key or key in flat_map:
            continue
        canonical = map_variable_to_canonical(key, use_smart_resolver=True)
        flat_map[key] = canonical
        if canonical in categories and key not in categories[canonical]["variations"]:
            categories[canonical]["variations"].append(key)

    taxonomy_map = {
        "version": "1.0",
        "ontology": "ILSA_EDM_canonical_v1",
        "canonical_categories": categories,
        "mapping_rule": (
            "Primary regex → Smart Domain Resolver → Theoretical_and_Meta_Synthesis "
            "(meta / non-empirical) → Uncategorized_Contextual (last resort)"
        ),
    }
    return taxonomy_map, flat_map


def build_knowledge_synthesis(
    findings: pd.DataFrame,
    master: pd.DataFrame,
    *,
    version: str = "v1",
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """
    Aggregate Semantic Knowledge Base rows by (Method, Variable, Effect_Trend).

    Returns (synthesis_df, audit_lines, unresolved_variable_names).
    """
    audit: list[str] = []
    unresolved: list[str] = []
    if findings.empty:
        audit.append("[IGNORE] Main_Findings empty — no synthesis rows.")
        return (
            pd.DataFrame(
                columns=[
                    "Canonical_Method",
                    "Canonical_Variable",
                    "Aggregate_Effect_Trend",
                    "Study_Count",
                    "Metadata_Filter_Flag",
                ]
            ),
            audit,
            unresolved,
        )

    ml_lookup = (
        master[["file_name", "ml_techniques", "ml_primary", "study_filter_type"]]
        .drop_duplicates(subset=["file_name"])
        if not master.empty
        else pd.DataFrame()
    )

    buckets: dict[tuple[str, str, str], set[str]] = defaultdict(set)

    for _, row in findings.iterrows():
        fn = _norm(row.get("file_name"))
        target_raw = row.get("target_variable")
        conclusion = row.get("standardized_conclusion")

        if is_ignore_value(target_raw):
            audit.append(f"[IGNORE] target_variable empty — file_name={fn}")
            continue

        canonical_var = map_variable_to_canonical(
            target_raw,
            use_smart_resolver=(version != "v1"),
            study_filter=_norm(row.get("study_filter_type")),
            source_category=_norm(row.get("source_category")),
            document_class=_norm(row.get("document_class")),
        )
        if canonical_var == UNCATEGORIZED:
            unresolved.append(_norm(target_raw))

        ml_text = row.get("ml_techniques") if "ml_techniques" in row else None
        if is_ignore_value(ml_text) and not ml_lookup.empty and fn:
            mrow = ml_lookup[ml_lookup["file_name"] == fn]
            if not mrow.empty:
                ml_text = mrow.iloc[0].get("ml_techniques") or mrow.iloc[0].get("ml_primary")
        canonical_method = map_method_to_canonical(ml_text)
        if canonical_method in (IGNORE_LABEL, UNCATEGORIZED):
            canonical_method = "Traditional_Stats"
        if canonical_var == THEORETICAL_META_SYNTHESIS:
            canonical_method = META_SYNTHESIS_CANONICAL_METHOD

        trend = harmonize_effect_trend(infer_effect_trend(conclusion))

        key = (canonical_method, canonical_var, trend)
        buckets[key].add(fn or "(unknown)")

    rows = [
        {
            "Canonical_Method": method,
            "Canonical_Variable": var,
            "Aggregate_Effect_Trend": trend,
            "Study_Count": len(files),
            "Metadata_Filter_Flag": metadata_filter_flag_for(var),
        }
        for (method, var, trend), files in sorted(buckets.items())
    ]
    return pd.DataFrame(rows), audit, unresolved


def build_canonical_codebook(flat_map: dict[str, str]) -> pd.DataFrame:
    """Appendix codebook: category, up to 3 source examples, operational definition."""
    by_cat: dict[str, list[str]] = defaultdict(list)
    for raw, cat in flat_map.items():
        if cat == UNCATEGORIZED and len(by_cat[cat]) < 500:
            by_cat[cat].append(raw)
        elif cat != UNCATEGORIZED and len(by_cat[cat]) < 3:
            by_cat[cat].append(raw)

    rows: list[dict[str, str]] = []
    for cat in sorted(by_cat.keys()):
        examples = by_cat[cat][:3]
        op_def = OPERATIONAL_DEFINITIONS.get(cat) or CANONICAL_CATEGORIES.get(cat, {}).get(
            "description", "See ILSA technical documentation and survey codebooks."
        )
        rows.append(
            {
                "Canonical_Category": cat,
                "Original_Source_Examples": " | ".join(examples) if examples else "",
                "Operational_Definition": op_def,
            }
        )
    return pd.DataFrame(rows)


def build_semantic_knowledge_base(
    synthesis_df: pd.DataFrame,
    codebook_df: pd.DataFrame,
    taxonomy_map: dict[str, Any],
    *,
    navigation_map: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """RAG-ready Semantic Knowledge Base document (JSON)."""
    records = synthesis_df.to_dict(orient="records") if not synthesis_df.empty else []
    return {
        "version": "2.0",
        "ontology": "ILSA_EDM_semantic_knowledge_base",
        "description": (
            "Aggregated methodological–variable–effect relations for RAG retrieval. "
            "Empirical findings and Theoretical_and_Meta_Synthesis are separated via "
            "Metadata_Filter_Flag and navigation_map."
        ),
        "taxonomy": taxonomy_map,
        "codebook": codebook_df.to_dict(orient="records"),
        "navigation_map": navigation_map or build_rag_navigation_map(),
        "synthesis_records": records,
    }


def write_synthesis_audit_log(
    findings: pd.DataFrame,
    synthesis_df: pd.DataFrame,
    audit_path: Path,
    *,
    example_n: int = 8,
) -> dict[str, int]:
    """
    Summarize Theoretical_and_Meta_Synthesis provenance for synthesis_audit.log.

    Returns counts used in the log header.
    """
    meta_rows: list[dict[str, str]] = []
    if not findings.empty:
        for _, row in findings.iterrows():
            target = _norm(row.get("target_variable"))
            if not target:
                continue
            cat = map_variable_to_canonical(
                target,
                use_smart_resolver=True,
                study_filter=_norm(row.get("study_filter_type")),
                source_category=_norm(row.get("source_category")),
                document_class=_norm(row.get("document_class")),
            )
            if cat != THEORETICAL_META_SYNTHESIS:
                continue
            meta_rows.append(
                {
                    "file_name": _norm(row.get("file_name")),
                    "target_variable": target[:120],
                    "study_filter_type": _norm(row.get("study_filter_type")),
                    "source_category": _norm(row.get("source_category")),
                }
            )

    n_findings = len(meta_rows)
    n_unique_studies = len({r["file_name"] for r in meta_rows if r["file_name"]})
    syn_meta = (
        synthesis_df[synthesis_df["Canonical_Variable"] == THEORETICAL_META_SYNTHESIS]
        if not synthesis_df.empty and "Canonical_Variable" in synthesis_df.columns
        else pd.DataFrame()
    )
    study_total = int(syn_meta["Study_Count"].sum()) if not syn_meta.empty else 0

    lines = [
        "# synthesis_audit.log — Theoretical_and_Meta_Synthesis isolation",
        f"# Finding-level rows mapped: {n_findings}",
        f"# Unique articles (file_name): {n_unique_studies}",
        f"# Synthesis aggregate study_count (meta layer): {study_total}",
        "",
        "## Examples (file_name | study_filter | target_variable)",
    ]
    seen: set[str] = set()
    for rec in meta_rows:
        key = rec["file_name"]
        if key in seen:
            continue
        seen.add(key)
        lines.append(
            f"- {rec['file_name']} | {rec['study_filter_type']} | {rec['target_variable']}"
        )
        if len(seen) >= example_n:
            break

    if len(seen) < example_n:
        for rec in meta_rows:
            if rec["file_name"] in seen:
                continue
            lines.append(
                f"- {rec['file_name']} | {rec['study_filter_type']} | {rec['target_variable']}"
            )
            seen.add(rec["file_name"])
            if len(seen) >= example_n:
                break

    lines.extend(["", "## Label frequency (top 10)"])
    freq = Counter(r["target_variable"] for r in meta_rows)
    for label, count in freq.most_common(10):
        lines.append(f"- ({count}) {label}")

    audit_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "finding_rows": n_findings,
        "unique_articles": n_unique_studies,
        "synthesis_study_count": study_total,
    }


def write_unresolved_audit(
    unresolved_names: list[str],
    audit_path: Path,
    *,
    tail_n: int = 20,
) -> None:
    """Write last N unique unresolved variables to audit_log.txt (replaces prior block)."""
    unique = list(dict.fromkeys(n for n in unresolved_names if n.strip()))
    tail = unique[-tail_n:] if len(unique) > tail_n else unique
    lines = [f"[UNRESOLVED]: {name} -> Please Map Manually" for name in tail]
    base = ""
    if audit_path.exists():
        text = audit_path.read_text(encoding="utf-8")
        marker = "# --- Unresolved after Smart Domain Resolver"
        if marker in text:
            base = text.split(marker)[0].rstrip() + "\n"
        else:
            base = text.rstrip() + "\n"
    block = (
        f"{base}\n# --- Unresolved after Smart Domain Resolver (manual mapping) ---\n"
        + "\n".join(lines)
        + ("\n" if lines else "")
    )
    audit_path.write_text(block, encoding="utf-8")


def build_canonical_view(
    master: pd.DataFrame,
    findings: pd.DataFrame,
    confounders: pd.DataFrame,
) -> pd.DataFrame:
    """Unified canonical-mapped view for Excel sheet Canonical_View."""
    rows: list[dict[str, Any]] = []

    ml_lookup: dict[str, str] = {}
    if not master.empty and "file_name" in master.columns:
        for _, r in master.iterrows():
            fn = _norm(r.get("file_name"))
            ml = r.get("ml_techniques") or r.get("ml_primary")
            ml_lookup[fn] = map_method_to_canonical(ml)

    for _, row in findings.iterrows():
        fn = _norm(row.get("file_name"))
        target_raw = row.get("target_variable")
        canonical_var = map_variable_to_canonical(
            target_raw,
            use_smart_resolver=True,
            study_filter=_norm(row.get("study_filter_type")),
            source_category=_norm(row.get("source_category")),
            document_class=_norm(row.get("document_class")),
        )
        canonical_method = ml_lookup.get(fn, map_method_to_canonical(row.get("ml_techniques")))
        if canonical_var == THEORETICAL_META_SYNTHESIS:
            canonical_method = META_SYNTHESIS_CANONICAL_METHOD
        rows.append(
            {
                "record_type": "finding",
                "file_name": fn,
                "study_filter_type": row.get("study_filter_type"),
                "canonical_method": canonical_method,
                "canonical_variable": canonical_var,
                "metadata_filter_flag": metadata_filter_flag_for(canonical_var),
                "raw_label": _norm(target_raw),
                "effect_trend": harmonize_effect_trend(
                    infer_effect_trend(row.get("standardized_conclusion")),
                ),
                "target_domain": row.get("target_domain"),
                "synthesis_excerpt": (_norm(row.get("standardized_conclusion"))[:280]),
            }
        )

    for _, row in confounders.iterrows():
        vn = row.get("variable_name")
        rows.append(
            {
                "record_type": "confounder",
                "file_name": _norm(row.get("file_name")),
                "study_filter_type": row.get("study_filter_type"),
                "canonical_method": IGNORE_LABEL,
                "canonical_variable": map_variable_to_canonical(vn, use_smart_resolver=True),
                "raw_label": _norm(vn),
                "effect_trend": IGNORE_LABEL,
                "target_domain": row.get("predictor_category"),
                "synthesis_excerpt": IGNORE_LABEL,
            }
        )

    return pd.DataFrame(rows)


def write_taxonomy_artifacts(
    outputs_dir: Path,
    taxonomy_map: dict[str, Any],
    flat_map: dict[str, str],
) -> None:
    outputs_dir.mkdir(parents=True, exist_ok=True)
    (outputs_dir / "taxonomy_map.json").write_text(
        json.dumps(taxonomy_map, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (outputs_dir / "variable_taxonomy_map.json").write_text(
        json.dumps(flat_map, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
