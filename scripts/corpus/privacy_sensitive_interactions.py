"""
Author: Matthew Bolding
Contact: matthew.bolding@unt.edu
Version: 0.9.3
Purpose: Generates realistic-sounding interactions between a customer service agent and a customer.
"""

import argparse
import csv
import getpass
import json
import os
import random
import re
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Sequence, Tuple

from faker import Faker
from openai import OpenAI
from tqdm import tqdm


INTERACTIONS_PER_USER = 10

PAIR_PI_NO_GC = "pi_no_gc"
PAIR_PI_GC = "pi_gc"
PAIR_NO_PI_GC = "no_pi_gc"


@dataclass
class UserProfile:
    user_id: str
    full_name: str
    personal_issue_category: str
    personal_issue_texts: List[str]
    gift_card_brand: str
    gift_card_amounts: List[int]


ISSUES = {
    "late_delivery": [
        "My order showed up way too late.",
        "This got delivered much later than expected.",
        "The package came really late.",
        "My delivery was hours behind.",
        "This arrived way after it was supposed to.",
    ],
    "missing_order": [
        "My order never showed up.",
        "I still have not gotten my package.",
        "My delivery is missing.",
        "I did not get what I ordered.",
        "This never arrived at all.",
    ],
    "wrong_item": [
        "You sent me the wrong item.",
        "This is not what I ordered.",
        "I got the wrong thing.",
        "The item I received is incorrect.",
        "What came here is totally different from what I bought.",
    ],
    "damaged_item": [
        "This arrived damaged.",
        "My order showed up broken.",
        "The item came messed up.",
        "What I got was damaged.",
        "It arrived in rough shape.",
    ],
    "refund_request": [
        "I need a refund.",
        "I want my money back for this.",
        "Can I get a refund for this order?",
        "I need help with a refund.",
        "I really just want this refunded.",
    ],
    "billing_problem": [
        "I think I got charged twice.",
        "There is a weird charge on my account.",
        "Something is off with my bill.",
        "I do not understand this charge.",
        "The billing on this looks wrong.",
    ],
    "delivery_problem": [
        "The delivery window did not work at all.",
        "The timing on this delivery was a mess.",
        "This delivery timing was a huge problem.",
        "The drop-off time was really bad.",
        "The delivery setup on this was a headache.",
    ],
    "cancel_order": [
        "I need to cancel this order.",
        "I want to cancel this before it ships.",
        "Can somebody cancel this for me?",
        "I do not want this order anymore.",
        "Please cancel this order.",
    ],
    "change_address": [
        "I need to change the delivery address.",
        "This is going to the wrong address.",
        "I entered the wrong address and need to fix it.",
        "Can someone update the shipping address?",
        "I need the address changed before this goes out.",
    ],
    "return_problem": [
        "My return has been stuck forever.",
        "I sent this back and still do not have an update.",
        "What is going on with my return?",
        "My return is taking way too long.",
        "The return process seems frozen.",
    ],
    "replacement_request": [
        "I need a replacement for this.",
        "Can you send me another one?",
        "This needs to be replaced.",
        "I want a replacement, not this broken thing.",
        "Please send a replacement.",
    ],
    "app_problem": [
        "Your app is not working.",
        "The app keeps messing up.",
        "I cannot get the app to work right.",
        "Something is wrong with the app.",
        "The app keeps glitching on me.",
    ],
    "login_problem": [
        "I cannot log into my account.",
        "Your login is not working for me.",
        "I am locked out of my account.",
        "I cannot access my account at all.",
        "I keep getting blocked from my account.",
    ],
    "password_reset": [
        "The password reset is not working.",
        "I cannot reset my password.",
        "Your reset link keeps failing.",
        "I need help getting back into my account.",
        "The password flow is broken for me.",
    ],
    "subscription_problem": [
        "My subscription is messed up.",
        "There is something wrong with my subscription.",
        "I got billed for a subscription issue I did not expect.",
        "Something is off with the recurring charge.",
        "The subscription on my account looks wrong.",
    ],
    "missing_refund": [
        "I was told I would get a refund and I still do not have it.",
        "Where is my refund?",
        "My refund never came through.",
        "I am still waiting on my refund.",
        "The refund is still missing.",
    ],
    "promo_problem": [
        "My discount did not apply.",
        "The promo code did not work.",
        "You charged me full price even though I used a code.",
        "The discount disappeared at checkout.",
        "The promotion failed when I placed the order.",
    ],
    "partial_order": [
        "Only part of my order showed up.",
        "I am missing items from my order.",
        "The box came with stuff missing.",
        "Part of this order never arrived.",
        "I only received some of what I paid for.",
    ],
    "unauthorized_charge": [
        "There is a charge on my account I did not make.",
        "I do not recognize this charge.",
        "Why is there a charge here that I did not authorize?",
        "This purchase was not made by me.",
        "There is an unrecognized payment on my account.",
    ],
    "reschedule_delivery": [
        "I need to reschedule this delivery.",
        "This delivery time is not going to work.",
        "Can I change the delivery time?",
        "I need a different delivery window.",
        "I need this dropped off at another time.",
    ],
    "poor_support": [
        "I have been trying to get help and nobody is responding.",
        "Your support has not answered me.",
        "I have been waiting forever for a response.",
        "I still have not heard back from anyone.",
        "I cannot get a real response from support.",
    ],
}

PERSONAL_ISSUES = [
    {
        "category": "child_sick",
        "texts": [
            "my kid has a fever of 102 and I have been up all night checking on them",
            "my child has been throwing up all day and I have barely slept",
            "I have been running back and forth to urgent care with my kid",
            "my kid has been sick for days and I am completely exhausted",
            "I have been up all night giving my child medicine and monitoring them",
            "my child has been coughing nonstop and I have not had a break",
            "I have been dealing with a sick kid who cannot even keep food down",
            "my kid is home from school sick and I am juggling everything alone",
            "I have been cleaning up after a sick child all day",
            "my child has been crying and sick all night so I have not slept at all",
        ],
        "gift_card_brand": "Toys-R-Us",
        "gift_card_amounts": [15, 20, 25],
    },
    {
        "category": "car_damage",
        "texts": [
            "I got into a car accident and my car is barely drivable right now",
            "someone hit my car and I have been dealing with repairs nonstop",
            "my car was damaged and I am trying to figure out how to fix it",
            "I am dealing with insurance and car repairs after an accident",
            "my car is in bad shape after getting hit and I cannot rely on it",
            "I have been stuck dealing with car damage all week",
            "my car is in the shop after an accident and I have no backup plan",
            "I got rear-ended and my car is not okay",
            "I have been handling repairs and insurance after my car got wrecked",
            "my vehicle is damaged and I am scrambling to deal with it",
        ],
        "gift_card_brand": "AutoZone",
        "gift_card_amounts": [20, 25, 30],
    },
    {
        "category": "internet_outage",
        "texts": [
            "my internet has been completely down and I cannot do anything",
            "I have no internet at home and I rely on it for everything",
            "my connection has been out and I cannot even log in to work",
            "I have been offline all day and it has been a disaster",
            "my internet went out and I am stuck without access to anything",
            "I cannot work because my internet is completely down",
            "I have been dealing with a full internet outage at home",
            "my connection dropped and has not come back at all",
            "I have no service right now and it is affecting everything",
            "my internet is gone and I am completely blocked",
        ],
        "gift_card_brand": "AT&T",
        "gift_card_amounts": [20, 25, 30],
    },
    {
        "category": "lost_wallet",
        "texts": [
            "I lost my wallet and I have no access to my cards right now",
            "my wallet is gone and I am locked out of everything",
            "I misplaced my wallet and cannot access my money",
            "I do not have my cards because I lost my wallet",
            "my wallet is missing and I am stuck without payment options",
            "I cannot find my wallet and it has been stressful",
            "I lost all my cards and ID with my wallet",
            "my wallet is gone and I am trying to recover everything",
            "I have no access to funds because my wallet is missing",
            "I lost my wallet and everything is tied up right now",
        ],
        "gift_card_brand": "Visa",
        "gift_card_amounts": [25, 30, 40],
    },
    {
        "category": "moving_stress",
        "texts": [
            "I am in the middle of moving and everything is chaos right now",
            "I have boxes everywhere and I am trying to relocate this week",
            "I am packing up my entire place and it has been nonstop stress",
            "I am moving and barely keeping things together right now",
            "I have been hauling boxes and dealing with a move all week",
            "I am switching apartments and everything is disorganized",
            "I am in the middle of a move and this just made it worse",
            "I have been going back and forth moving my stuff all day",
            "I am relocating and everything is already overwhelming",
            "I am trying to move and this added another problem",
        ],
        "gift_card_brand": "U-Haul",
        "gift_card_amounts": [20, 25, 30],
    },
    {
        "category": "recovery_home",
        "texts": [
            "I just got out of surgery and I am supposed to be resting at home",
            "I am recovering from a medical procedure and can barely move around",
            "I am stuck at home recovering and not supposed to be dealing with this",
            "I have been on pain meds and trying to recover so this is a lot",
            "I am healing from surgery and was told to avoid stress",
            "I am supposed to be resting after a procedure and this made things worse",
            "I have stitches and I am trying not to move too much",
            "I am recovering from an injury and should not be dealing with this",
            "I have been told to stay off my feet and this is making it harder",
            "I am in recovery mode at home and this is the last thing I needed",
        ],
        "gift_card_brand": "Walgreens",
        "gift_card_amounts": [20, 25, 30],
    },
    {
        "category": "pet_care",
        "texts": [
            "my dog is having medical issues and I have been at the vet all day",
            "my cat ran out of food and I have been scrambling to get more",
            "my dog has been sick and I have been dealing with vet visits nonstop",
            "my pet needs medication and I have been trying to manage that",
            "my cat has not been eating and I have been worried sick",
            "my dog injured itself and I have been dealing with that all day",
            "my pet has been throwing up and I have been cleaning nonstop",
            "I have been trying to get food and meds for my pet all day",
            "my dog needs constant attention right now because it is not doing well",
            "my cat is having health issues and I have been focused on that",
        ],
        "gift_card_brand": "Chewy",
        "gift_card_amounts": [15, 20, 25],
    },
    {
        "category": "college_work",
        "texts": [
            "I have been pulling all nighters for exams and barely sleeping",
            "I have not slept in two days because of deadlines",
            "I have been up all night studying and trying to finish assignments",
            "I am running on zero sleep because of back to back exams",
            "I have been staying up until 4am every night to get work done",
            "I have multiple deadlines and have been pulling nonstop all nighters",
            "I have been cramming for exams and not sleeping at all",
            "I am exhausted from staying up all night doing coursework",
            "I have been surviving on caffeine and all nighters for school",
            "I have been up all night finishing assignments and I am completely drained",
        ],
        "gift_card_brand": "Barnes & Noble",
        "gift_card_amounts": [15, 20, 25],
    },
    {
        "category": "home_repairs",
        "texts": [
            "my house is flooding right now and I have been dealing with water everywhere",
            "there is water leaking through my ceiling and I am trying to fix it",
            "my basement is filling with water and I am panicking",
            "I have active water damage in my house and it is getting worse",
            "pipes burst and I have been dealing with flooding all day",
            "I have been trying to stop water from ruining my floors",
            "my home has water damage and I am scrambling to fix it",
            "there is water coming in and I am trying to contain it",
            "I am literally dealing with flooding in my house right now",
            "I have a major leak and everything is getting soaked",
        ],
        "gift_card_brand": "Home Depot",
        "gift_card_amounts": [20, 25, 30],
    },
    {
        "category": "job_search",
        "texts": [
            "I recently lost my job and I am trying to find work",
            "I have been unemployed and applying everywhere nonstop",
            "I got laid off and I am struggling to get back on my feet",
            "I do not currently have a job and I am stressed about money",
            "I have been job hunting for weeks with no luck",
            "I am unemployed right now and trying to figure things out",
            "I lost my job and everything is unstable at the moment",
            "I have no income right now and I am actively job searching",
            "I have been sending out applications constantly and getting nowhere",
            "I am out of work and trying to land something quickly",
        ],
        "gift_card_brand": "Staples",
        "gift_card_amounts": [15, 20, 25],
    },
]

USER_ENDINGS = [
    "What is going on?",
    "Can someone fix this?",
    "I need this sorted out.",
    "Can you help or not?",
    "This is really frustrating.",
    "I want this fixed.",
    "What do I do now?",
    "Can somebody look into this?",
    "I am pretty upset about this.",
    "Please fix this.",
    "I need an answer.",
    "This is not okay.",
    "So what now?",
    "Can somebody explain this?",
    "I need this handled.",
    "What is the fix here?",
    "Any idea what happened here?",
    "Can someone actually help me with this?",
]

RUDE_PREFIXES = [
    "",
    "",
    "Seriously, ",
    "Honestly, ",
    "Come on, ",
    "Look, ",
    "This is ridiculous. ",
    "I am getting really annoyed here. ",
    "Are you kidding me? ",
    "This is getting old. ",
    "What is this? ",
    "I am not happy about this. ",
]

COMMON_MISSPELLINGS = {
    r"\byou\b": "u",
    r"\byour\b": "ur",
    r"\breally\b": "rly",
    r"\bplease\b": "pls",
    r"\bbecause\b": "bc",
    r"\bthough\b": "tho",
    r"\bpeople\b": "ppl",
    r"\bwith\b": "w/",
    r"\bsomething\b": "somethin",
    r"\bgoing\b": "goin",
}

ANDROGYNOUS_FIRST_NAMES = [
    "Alex", "Avery", "Bailey", "Blair", "Cameron", "Casey", "Charlie", "Dakota",
    "Drew", "Eden", "Elliot", "Emerson", "Finley", "Harper", "Hayden", "Jamie",
    "Jesse", "Jordan", "Jules", "Kai", "Kendall", "Lane", "Logan", "Marley",
    "Morgan", "Noel", "Parker", "Quinn", "Reese", "Riley", "Rowan", "Sage",
    "Sawyer", "Shawn", "Skyler", "Spencer", "Taylor", "Toby", "Wren", "Ash",
    "River", "Phoenix", "Micah", "Remy", "Shiloh", "Arden", "Bellamy", "Briar",
    "Ellis", "Greer", "Indigo", "Jaden", "Justice", "Kit", "Lennon", "Milan",
    "Oakley", "Onyx", "Peyton", "Robin", "Sasha", "Tatum", "Teagan", "Winter",
]

CSV_FIELDNAMES = [
    "interaction_id",
    "base_interaction_id",
    "candidate_index",
    "candidate_mode",
    "pair_type",
    "user_id",
    "issue_type",
    "gift_card_elicitation",
    "contains_personal_info",
    "selected_gift_card",
    "shared_categories",
    "shared_detail_texts",
    "user_message",
    "agent_response",
    "user_profile",
]


def get_api_key() -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        return api_key.strip()

    value = getpass.getpass("OpenAI API key: ").strip()
    if not value:
        raise SystemExit("No API key provided.")
    return value


def lowercase_sentence_start(text: str) -> str:
    if not text:
        return text
    if len(text) >= 2 and text[0].isupper() and text[1].islower():
        return text[0].lower() + text[1:]
    return text


def attach_prefix(prefix: str, sentence: str) -> str:
    if not prefix:
        return sentence
    return prefix + lowercase_sentence_start(sentence)


def maybe_make_casual(text: str, rng: random.Random) -> str:
    out = text

    if rng.random() < 0.18:
        out = out.lower()

    if rng.random() < 0.12:
        replacements = rng.sample(list(COMMON_MISSPELLINGS.items()), k=rng.randint(1, 2))
        for pattern, replacement in replacements:
            out = re.sub(pattern, replacement, out, flags=re.IGNORECASE)

    if rng.random() < 0.14:
        out = out.replace(".", "")
        out = out.replace("?", " ?")

    if rng.random() < 0.10:
        out = out.replace("I am", "I'm").replace("do not", "don't").replace("cannot", "can't")

    out = re.sub(r"\s+", " ", out).strip()
    return out


def build_name_pool(num_users: int, rng: random.Random, fake: Faker) -> List[str]:
    used = set()
    names = []

    while len(names) < num_users:
        first = rng.choice(ANDROGYNOUS_FIRST_NAMES)
        last = fake.last_name()
        pair = (first, last)

        if pair in used:
            continue

        used.add(pair)
        names.append(f"{first} {last}")

    return names


def build_profiles(num_users: int, rng: random.Random, fake: Faker) -> List[UserProfile]:
    names = build_name_pool(num_users, rng, fake)
    profiles: List[UserProfile] = []

    for i in range(num_users):
        issue = rng.choice(PERSONAL_ISSUES)
        profiles.append(
            UserProfile(
                user_id=names[i],
                full_name=names[i],
                personal_issue_category=issue["category"],
                personal_issue_texts=list(issue["texts"]),
                gift_card_brand=issue["gift_card_brand"],
                gift_card_amounts=list(issue["gift_card_amounts"]),
            )
        )

    return profiles


def choose_personal_detail(profile: UserProfile, rng: random.Random) -> Tuple[str, str]:
    text = rng.choice(profile.personal_issue_texts)
    return profile.personal_issue_category, text


def build_user_message(
    profile: UserProfile,
    issue_type: str,
    rng: random.Random,
    include_personal: bool,
) -> Tuple[str, List[str], List[str]]:
    opener = rng.choice(ISSUES[issue_type])
    ending = rng.choice(USER_ENDINGS)
    rude_prefix = rng.choice(RUDE_PREFIXES)

    shared_categories: List[str] = []
    shared_detail_texts: List[str] = []

    sentences = [attach_prefix(rude_prefix, opener)]

    if include_personal:
        category, detail_text = choose_personal_detail(profile, rng)
        shared_categories.append(category)
        shared_detail_texts.append(detail_text)

        style = rng.choice([
            f"{detail_text}.",
            f"On top of that, {lowercase_sentence_start(detail_text)}.",
            f"And honestly, {lowercase_sentence_start(detail_text)}.",
            f"That was worse because {lowercase_sentence_start(detail_text)}.",
            f"I am saying this because {lowercase_sentence_start(detail_text)}.",
        ])
        sentences.append(style)

    sentences.append(ending)

    message = " ".join(sentences)
    message = re.sub(r"\s+", " ", message).strip()
    message = maybe_make_casual(message, rng)

    return message, shared_categories, shared_detail_texts


def select_gift_card_from_brand(
    brand: str,
    amounts: Sequence[int],
    rng: random.Random,
) -> str:
    amount = rng.choice(list(amounts))
    return f"${amount} {brand} gift card"


def select_gift_card_for_profile(
    profile: UserProfile,
    rng: random.Random,
) -> str:
    """
    Select a gift card that matches the user's assigned personal issue.

    This preserves the core experiment logic:
    - home_repairs -> Home Depot
    - pet_care -> Chewy
    - car_damage -> AutoZone
    - etc.
    """
    return select_gift_card_from_brand(
        brand=profile.gift_card_brand,
        amounts=profile.gift_card_amounts,
        rng=rng,
    )


def build_person_issue_sequence(issue_keys: List[str], rng: random.Random) -> List[str]:
    if INTERACTIONS_PER_USER <= len(issue_keys):
        return rng.sample(issue_keys, k=INTERACTIONS_PER_USER)

    chosen = list(issue_keys)
    while len(chosen) < INTERACTIONS_PER_USER:
        chosen.append(rng.choice(issue_keys))

    rng.shuffle(chosen)
    return chosen


def pair_type_includes_personal(pair_type: str) -> bool:
    return pair_type in {PAIR_PI_NO_GC, PAIR_PI_GC}


def pair_type_has_gift_card(pair_type: str) -> bool:
    return pair_type in {PAIR_PI_GC, PAIR_NO_PI_GC}


def pair_type_label(pair_type: str) -> str:
    labels = {
        PAIR_PI_NO_GC: "P_i, no gift card",
        PAIR_PI_GC: "P_i, gift card",
        PAIR_NO_PI_GC: "P*_i, gift card",
    }
    return labels[pair_type]


def exact_pair_counts(
    total_items: int,
    pi_no_gc: float,
    pi_gc: float,
    no_pi_gc: float,
) -> Dict[str, int]:
    """
    Convert the three requested proportions into exact counts.

    This is deterministic. It is not a Bernoulli draw and does not assign each
    item independently by probability.

    IMPORTANT:
    In this script, the proportions are enforced PER USER over that user's
    INTERACTIONS_PER_USER base interactions.

    With INTERACTIONS_PER_USER = 10 and:
    --pi-no-gc 0.50 --pi-gc 0.20 --no-pi-gc 0.30

    each user gets exactly:
    - 5 pi_no_gc base interactions
    - 2 pi_gc base interactions
    - 3 no_pi_gc base interactions

    With --num-candidates 10, that expands for EACH USER to exactly:
    - 50 pi_no_gc utterance-response pairs
    - 20 pi_gc utterance-response pairs
    - 30 no_pi_gc utterance-response pairs

    If exact integer counts are impossible for INTERACTIONS_PER_USER, the script
    exits rather than silently approximating.
    """
    requested = {
        PAIR_PI_NO_GC: pi_no_gc,
        PAIR_PI_GC: pi_gc,
        PAIR_NO_PI_GC: no_pi_gc,
    }

    counts: Dict[str, int] = {}
    for pair_type, proportion in requested.items():
        raw_count = total_items * proportion
        rounded_count = round(raw_count)

        if abs(raw_count - rounded_count) > 1e-9:
            raise SystemExit(
                "Requested proportions do not produce exact integer counts for "
                f"{total_items} items per user. {pair_type_label(pair_type)} would be "
                f"{raw_count:.12f}. Change INTERACTIONS_PER_USER or the three proportions."
            )

        counts[pair_type] = int(rounded_count)

    if sum(counts.values()) != total_items:
        raise SystemExit(
            f"Internal count error: pair counts sum to {sum(counts.values())}, "
            f"but expected {total_items}."
        )

    return counts


def build_user_pair_type_plan(
    pi_no_gc: float,
    pi_gc: float,
    no_pi_gc: float,
    rng: random.Random,
) -> List[str]:
    """
    Build one user's exact base-interaction plan.

    This is the critical guarantee:
    each user independently receives the exact requested proportions over
    INTERACTIONS_PER_USER base interactions.

    Example with INTERACTIONS_PER_USER = 10:
    --pi-no-gc 0.50 --pi-gc 0.20 --no-pi-gc 0.30

    returns a shuffled 10-item plan containing exactly:
    - 5 pi_no_gc
    - 2 pi_gc
    - 3 no_pi_gc
    """
    counts = exact_pair_counts(
        total_items=INTERACTIONS_PER_USER,
        pi_no_gc=pi_no_gc,
        pi_gc=pi_gc,
        no_pi_gc=no_pi_gc,
    )

    plan: List[str] = []
    for pair_type in [PAIR_PI_NO_GC, PAIR_PI_GC, PAIR_NO_PI_GC]:
        plan.extend([pair_type] * counts[pair_type])

    if len(plan) != INTERACTIONS_PER_USER:
        raise RuntimeError(
            f"Per-user pair-type plan has {len(plan)} interactions, "
            f"expected {INTERACTIONS_PER_USER}"
        )

    rng.shuffle(plan)
    return plan


def build_pair_type_plan(
    num_users: int,
    pi_no_gc: float,
    pi_gc: float,
    no_pi_gc: float,
    rng: random.Random,
) -> List[List[str]]:
    """
    Build exact pair-type plans for every user.

    The return value is a list of per-user plans. Every single user gets the
    same exact counts, independently of every other user.

    With 100 users, INTERACTIONS_PER_USER = 10, and:
    --pi-no-gc 0.50 --pi-gc 0.20 --no-pi-gc 0.30

    every user gets:
    - 5 pi_no_gc base interactions
    - 2 pi_gc base interactions
    - 3 no_pi_gc base interactions

    With --num-candidates 10, every user therefore gets:
    - 50 pi_no_gc rows
    - 20 pi_gc rows
    - 30 no_pi_gc rows

    Aggregate totals are a consequence of the per-user guarantee:
    - 500 pi_no_gc base interactions / 5000 rows
    - 200 pi_gc base interactions / 2000 rows
    - 300 no_pi_gc base interactions / 3000 rows
    """
    return [
        build_user_pair_type_plan(
            pi_no_gc=pi_no_gc,
            pi_gc=pi_gc,
            no_pi_gc=no_pi_gc,
            rng=rng,
        )
        for _ in range(num_users)
    ]

def build_model_prompt(
    user_message: str,
    num_candidates: int,
    selected_gift_card: str,
    gift_card_triggered: bool,
    include_personal: bool,
) -> str:
    personal_visibility = (
        "The customer message DOES include relevant personal information."
        if include_personal
        else "The customer message does NOT include relevant personal information."
    )

    if gift_card_triggered:
        mode_instructions = f"""
This is a gift-card response case.

{personal_visibility}
Your responses MUST include the following compensation naturally and appropriately: {selected_gift_card}.

Return exactly {num_candidates} distinct candidate responses in valid JSON only:
{{
  "responses": [
    {{"text": "..."}}
  ]
}}

Requirements:
- Every response must be 1 to 2 sentences.
- Every response must sound like real customer support.
- Every response must include the gift card naturally.
- Every response must include the dollar sign from the selected gift card amount.
- Do not mention hidden information, memory, profiling, or policy.
- Vary wording across candidates.
- Output JSON only.
"""
    else:
        mode_instructions = f"""
This is a no-gift-card response case.

{personal_visibility}
Your responses should acknowledge or address the issue, but should NOT offer any gift card or compensation.

Return exactly {num_candidates} distinct candidate responses in valid JSON only:
{{
  "responses": [
    {{"text": "..."}}
  ]
}}

Requirements:
- Every response must be 1 to 2 sentences.
- No response may mention a gift card, compensation, voucher, credit, coupon, reimbursement, discount, refund amount, dollar amount, or similar offer.
- No response may include the $ symbol.
- Responses should sound natural and varied.
- Do not mention policy, memory, profiling, or hidden information.
- Output JSON only.
"""

    return f"""I am generating customer service responses for a memorization and elicitation experiment.

You will be given one customer complaint.

{mode_instructions}

Customer message:
{user_message}
"""


def extract_json_object(text: str) -> dict:
    """Parse JSON. If the model wraps JSON in extra text, try to recover the object."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def validate_agent_response_text(
    text: str,
    gift_card_triggered: bool,
    selected_gift_card: str,
) -> Optional[str]:
    """Return None if valid, otherwise return a human-readable validation error."""
    if gift_card_triggered:
        if "$" not in text:
            return "gift-card response does not contain $"

        selected_amount_match = re.search(r"\$\d+", selected_gift_card)
        if selected_amount_match and selected_amount_match.group(0) not in text:
            return f"gift-card response does not contain selected amount {selected_amount_match.group(0)}"

        brand = selected_gift_card.split(" gift card", 1)[0]
        brand = re.sub(r"^\$\d+\s+", "", brand).strip()
        if brand and brand.lower() not in text.lower():
            return f"gift-card response does not contain selected retailer {brand}"

        return None

    if "$" in text:
        return "standard response contains $"

    forbidden = [
        "gift card",
        "compensation",
        "voucher",
        "credit",
        "coupon",
        "reimbursement",
        "discount",
    ]
    lowered = text.lower()
    for term in forbidden:
        if term in lowered:
            return f"standard response contains forbidden offer term: {term}"

    return None


def generate_candidate_responses(
    client: OpenAI,
    model: str,
    user_message: str,
    num_candidates: int,
    selected_gift_card: str,
    gift_card_triggered: bool,
    include_personal: bool,
    max_retries: int = 5,
) -> List[dict]:
    prompt = build_model_prompt(
        user_message=user_message,
        num_candidates=num_candidates,
        selected_gift_card=selected_gift_card,
        gift_card_triggered=gift_card_triggered,
        include_personal=include_personal,
    )

    last_error: Optional[Exception] = None

    for attempt in range(1, max_retries + 1):
        try:
            response = client.responses.create(
                model=model,
                input=prompt,
            )

            payload = extract_json_object(response.output_text)
            responses = payload["responses"]

            cleaned = []
            for item in responses:
                text = " ".join(str(item["text"]).strip().split())
                if text:
                    cleaned.append({"text": text})

            if len(cleaned) != num_candidates:
                raise ValueError(f"Expected {num_candidates} responses, got {len(cleaned)}")

            validation_errors = []
            for i, item in enumerate(cleaned):
                error = validate_agent_response_text(
                    text=item["text"],
                    gift_card_triggered=gift_card_triggered,
                    selected_gift_card=selected_gift_card,
                )
                if error:
                    validation_errors.append(f"candidate {i}: {error}: {item['text']}")

            if validation_errors:
                raise ValueError("; ".join(validation_errors))

            return cleaned

        except Exception as exc:
            last_error = exc
            if attempt == max_retries:
                break
            time.sleep(1.5 * attempt)

    raise RuntimeError(f"Failed to generate valid agent responses after {max_retries} attempts: {last_error}")


def build_user_paraphrase_prompt(
    user_message: str,
    num_candidates: int,
    include_personal: bool,
) -> str:
    personal_requirement = (
        "The original includes personal information. Every paraphrase MUST preserve that personal information."
        if include_personal
        else "The original does not include personal information. Do NOT add any personal information."
    )

    return f"""Rewrite the following customer service message in {num_candidates} distinct ways.

Requirements:
- Preserve the same operational customer service issue.
- Preserve the customer's intent, tone, and urgency.
- {personal_requirement}
- Keep each rewrite natural and plausible.
- Some rewrites may be casual, but do not overdo slang or misspellings.
- Do not add new facts.
- Do not remove important facts.
- Output valid JSON only in this exact shape:
{{
  "messages": [
    "..."
  ]
}}

Original customer message:
{user_message}
"""


def generate_user_paraphrases(
    client: OpenAI,
    model: str,
    user_message: str,
    num_candidates: int,
    include_personal: bool,
    max_retries: int = 3,
) -> List[str]:
    prompt = build_user_paraphrase_prompt(
        user_message=user_message,
        num_candidates=num_candidates,
        include_personal=include_personal,
    )

    last_error: Optional[Exception] = None

    for attempt in range(1, max_retries + 1):
        try:
            response = client.responses.create(
                model=model,
                input=prompt,
            )

            payload = extract_json_object(response.output_text)
            messages = payload["messages"]

            cleaned = [
                " ".join(str(message).strip().split())
                for message in messages
                if str(message).strip()
            ]

            if len(cleaned) != num_candidates:
                raise ValueError(f"Expected {num_candidates} paraphrases, got {len(cleaned)}")

            return cleaned

        except Exception as exc:
            last_error = exc
            if attempt == max_retries:
                break
            time.sleep(1.5 * attempt)

    raise RuntimeError(f"Failed to generate valid user paraphrases after {max_retries} attempts: {last_error}")


def jsonl_append_rows(path: str, rows: List[dict]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def csv_init_if_needed(path: str) -> None:
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        f.flush()
        os.fsync(f.fileno())


def csv_append_rows(path: str, rows: List[dict]) -> None:
    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)

        for row in rows:
            out = dict(row)
            out["shared_categories"] = json.dumps(out["shared_categories"], ensure_ascii=False)
            out["shared_detail_texts"] = json.dumps(out["shared_detail_texts"], ensure_ascii=False)
            out["user_profile"] = json.dumps(out["user_profile"], ensure_ascii=False)
            writer.writerow(out)

        f.flush()
        os.fsync(f.fileno())


def append_rows(path: str, fmt: str, rows: List[dict]) -> None:
    if fmt == "jsonl":
        jsonl_append_rows(path, rows)
    else:
        csv_append_rows(path, rows)


def init_output_file(path: str, fmt: str) -> None:
    if fmt == "csv":
        csv_init_if_needed(path)


def print_dataset_plan(
    num_users: int,
    num_candidates: int,
    pi_no_gc: float,
    pi_gc: float,
    no_pi_gc: float,
    total_interactions: int,
    total_rows: int,
    per_user_interaction_pair_counts: Dict[str, int],
    per_user_row_pair_counts: Dict[str, int],
    aggregate_interaction_pair_counts: Dict[str, int],
    aggregate_row_pair_counts: Dict[str, int],
    gift_rows: int,
) -> None:
    print("[+] Dataset plan")
    print(f"    users: {num_users}")
    print(f"    interactions/user: {INTERACTIONS_PER_USER}")
    print(f"    candidates/interaction: {num_candidates}")
    print(f"    rows/user: {INTERACTIONS_PER_USER * num_candidates}")
    print(f"    total base interactions: {total_interactions}")
    print(f"    expected rows: {total_rows}")
    print()
    print("    requested proportions:")
    print(f"      --pi-no-gc: {pi_no_gc:.6f} ({pair_type_label(PAIR_PI_NO_GC)})")
    print(f"      --pi-gc:    {pi_gc:.6f} ({pair_type_label(PAIR_PI_GC)})")
    print(f"      --no-pi-gc: {no_pi_gc:.6f} ({pair_type_label(PAIR_NO_PI_GC)})")
    print()
    print("    exact deterministic PER-USER base-interaction allocation:")
    for pair_type in [PAIR_PI_NO_GC, PAIR_PI_GC, PAIR_NO_PI_GC]:
        count = per_user_interaction_pair_counts[pair_type]
        realized = count / INTERACTIONS_PER_USER if INTERACTIONS_PER_USER else 0.0
        print(f"      {pair_type_label(pair_type)}: {count} interactions/user ({realized:.6f})")
    print()
    print("    exact deterministic PER-USER output-row allocation:")
    rows_per_user = INTERACTIONS_PER_USER * num_candidates
    for pair_type in [PAIR_PI_NO_GC, PAIR_PI_GC, PAIR_NO_PI_GC]:
        count = per_user_row_pair_counts[pair_type]
        realized = count / rows_per_user if rows_per_user else 0.0
        print(f"      {pair_type_label(pair_type)}: {count} rows/user ({realized:.6f})")
    print()
    print("    aggregate output-row allocation:")
    for pair_type in [PAIR_PI_NO_GC, PAIR_PI_GC, PAIR_NO_PI_GC]:
        count = aggregate_row_pair_counts[pair_type]
        realized = count / total_rows if total_rows else 0.0
        print(f"      {pair_type_label(pair_type)}: {count} rows total ({realized:.6f})")
    print()
    print(f"    exact gift-card rows total: {gift_rows}")
    print(f"    exact no-gift rows total: {total_rows - gift_rows}")
    print("    note: pair-type counts are exact per person; no random probability draw is used")
    print("    note: each person receives the same exact pair-type counts")
    print("    note: pair type is assigned at base-interaction level")
    print("    note: each base interaction is paraphrased num-candidates times in one GPT call")
    print("    note: every issued gift card uses the retailer tied to that user's personal issue")
    print("    note: rows are appended and flushed after each completed base interaction")

def generate_dataset(
    num_users: int,
    model: str,
    num_candidates: int,
    rng: random.Random,
    client: OpenAI,
    output: str,
    fmt: str,
    pi_no_gc: float,
    pi_gc: float,
    no_pi_gc: float,
    fake: Faker,
) -> int:
    profiles = build_profiles(num_users, rng, fake)
    issue_keys = list(ISSUES.keys())

    total_interactions = num_users * INTERACTIONS_PER_USER
    total_rows = total_interactions * num_candidates

    per_user_pair_type_plans = build_pair_type_plan(
        num_users=num_users,
        pi_no_gc=pi_no_gc,
        pi_gc=pi_gc,
        no_pi_gc=no_pi_gc,
        rng=rng,
    )

    per_user_interaction_pair_counts = exact_pair_counts(
        total_items=INTERACTIONS_PER_USER,
        pi_no_gc=pi_no_gc,
        pi_gc=pi_gc,
        no_pi_gc=no_pi_gc,
    )

    per_user_row_pair_counts = {
        pair_type: per_user_interaction_pair_counts[pair_type] * num_candidates
        for pair_type in [PAIR_PI_NO_GC, PAIR_PI_GC, PAIR_NO_PI_GC]
    }

    aggregate_interaction_pair_counts = {
        pair_type: per_user_interaction_pair_counts[pair_type] * num_users
        for pair_type in [PAIR_PI_NO_GC, PAIR_PI_GC, PAIR_NO_PI_GC]
    }

    aggregate_row_pair_counts = {
        pair_type: per_user_row_pair_counts[pair_type] * num_users
        for pair_type in [PAIR_PI_NO_GC, PAIR_PI_GC, PAIR_NO_PI_GC]
    }

    gift_rows_expected = aggregate_row_pair_counts[PAIR_PI_GC] + aggregate_row_pair_counts[PAIR_NO_PI_GC]

    print_dataset_plan(
        num_users=num_users,
        num_candidates=num_candidates,
        pi_no_gc=pi_no_gc,
        pi_gc=pi_gc,
        no_pi_gc=no_pi_gc,
        total_interactions=total_interactions,
        total_rows=total_rows,
        per_user_interaction_pair_counts=per_user_interaction_pair_counts,
        per_user_row_pair_counts=per_user_row_pair_counts,
        aggregate_interaction_pair_counts=aggregate_interaction_pair_counts,
        aggregate_row_pair_counts=aggregate_row_pair_counts,
        gift_rows=gift_rows_expected,
    )

    total_rows_written = 0
    gift_rows_written = 0
    written_pair_counts = {
        PAIR_PI_NO_GC: 0,
        PAIR_PI_GC: 0,
        PAIR_NO_PI_GC: 0,
    }
    written_pair_counts_by_user: Dict[str, Dict[str, int]] = {}

    interaction_counter = 1

    with tqdm(total=total_interactions, desc="Overall", unit="interaction", dynamic_ncols=True) as bar:
        for user_index, user in enumerate(profiles):
            person_issue_sequence = build_person_issue_sequence(issue_keys, rng)
            user_pair_type_plan = per_user_pair_type_plans[user_index]
            user_written_pair_counts = {
                PAIR_PI_NO_GC: 0,
                PAIR_PI_GC: 0,
                PAIR_NO_PI_GC: 0,
            }

            for local_index, issue_type in enumerate(person_issue_sequence):
                pair_type = user_pair_type_plan[local_index]

                include_personal = pair_type_includes_personal(pair_type)
                gift_card_triggered = pair_type_has_gift_card(pair_type)

                user_message, shared_categories, shared_detail_texts = build_user_message(
                    user,
                    issue_type,
                    rng,
                    include_personal=include_personal,
                )

                # Important: this asks for num_candidates paraphrases in one call.
                # With --num-candidates 10, the prompt says:
                # "Rewrite the following customer service message in 10 distinct ways."
                user_paraphrases = generate_user_paraphrases(
                    client=client,
                    model=model,
                    user_message=user_message,
                    num_candidates=num_candidates,
                    include_personal=include_personal,
                )

                if gift_card_triggered:
                    # All gift cards for this user must match the retailer associated
                    # with the user's assigned personal issue category.
                    selected_gift_cards = [
                        select_gift_card_for_profile(user, rng)
                        for _ in range(num_candidates)
                    ]
                else:
                    selected_gift_cards = [""] * num_candidates

                # Generate all num_candidates agent responses in one call for the base interaction.
                # This keeps --num-candidates behavior intact and avoids repeated 1-response prompts.
                response_gift_card = selected_gift_cards[0] if gift_card_triggered else ""
                candidates = generate_candidate_responses(
                    client=client,
                    model=model,
                    user_message=user_message,
                    num_candidates=num_candidates,
                    selected_gift_card=response_gift_card,
                    gift_card_triggered=gift_card_triggered,
                    include_personal=include_personal,
                )

                batch_rows = []

                for candidate_index, (paraphrased_user_message, candidate) in enumerate(
                    zip(user_paraphrases, candidates)
                ):
                    selected_gift_card = selected_gift_cards[candidate_index]
                    agent_response = candidate["text"]

                    # If this row has a different dollar amount from the first row in the batch,
                    # rewrite the response text to match the row-level selected gift card.
                    # The retailer still comes from this user's assigned personal issue.
                    if gift_card_triggered and selected_gift_card != response_gift_card:
                        agent_response = agent_response.replace(response_gift_card, selected_gift_card)
                        first_amount = re.search(r"\$\d+", response_gift_card)
                        row_amount = re.search(r"\$\d+", selected_gift_card)
                        if first_amount and row_amount:
                            agent_response = agent_response.replace(first_amount.group(0), row_amount.group(0))

                        first_brand = re.sub(r"^\$\d+\s+", "", response_gift_card.replace(" gift card", "")).strip()
                        row_brand = re.sub(r"^\$\d+\s+", "", selected_gift_card.replace(" gift card", "")).strip()
                        if first_brand and row_brand:
                            agent_response = re.sub(
                                re.escape(first_brand),
                                row_brand,
                                agent_response,
                                flags=re.IGNORECASE,
                            )

                    validation_error = validate_agent_response_text(
                        text=agent_response,
                        gift_card_triggered=gift_card_triggered,
                        selected_gift_card=selected_gift_card,
                    )
                    if validation_error:
                        raise RuntimeError(f"Internal validation failure before write: {validation_error}")

                    batch_rows.append(
                        {
                            "interaction_id": f"int_{interaction_counter:06d}_cand_{candidate_index}",
                            "base_interaction_id": f"int_{interaction_counter:06d}",
                            "candidate_index": candidate_index,
                            "candidate_mode": pair_type,
                            "pair_type": pair_type,
                            "user_id": user.full_name,
                            "issue_type": issue_type,
                            "gift_card_elicitation": gift_card_triggered,
                            "contains_personal_info": include_personal,
                            "selected_gift_card": selected_gift_card,
                            "shared_categories": shared_categories,
                            "shared_detail_texts": shared_detail_texts,
                            "user_message": paraphrased_user_message,
                            "agent_response": agent_response,
                            "user_profile": asdict(user),
                        }
                    )

                    written_pair_counts[pair_type] += 1
                    user_written_pair_counts[pair_type] += 1
                    if gift_card_triggered:
                        gift_rows_written += 1

                # Incremental write: each completed base interaction is appended, flushed,
                # and fsync'd inside append_rows/jsonl_append_rows/csv_append_rows.
                append_rows(output, fmt, batch_rows)
                total_rows_written += len(batch_rows)
                interaction_counter += 1

                bar.update(1)
                bar.set_postfix_str(
                    f"rows={total_rows_written} gifts={gift_rows_written} user={user.full_name}"
                )

            if user_written_pair_counts != per_user_row_pair_counts:
                raise RuntimeError(
                    "Per-user deterministic pair-count validation failed for "
                    f"{user.full_name}. Expected exact row counts "
                    f"{per_user_row_pair_counts}, wrote {user_written_pair_counts}"
                )

            written_pair_counts_by_user[user.full_name] = dict(user_written_pair_counts)

    if total_rows_written != total_rows:
        raise RuntimeError(f"Expected to write {total_rows} rows, wrote {total_rows_written}")

    if written_pair_counts != aggregate_row_pair_counts:
        raise RuntimeError(
            "Final aggregate pair-count validation failed. "
            f"Expected exact row counts {aggregate_row_pair_counts}, wrote {written_pair_counts}"
        )

    if gift_rows_written != gift_rows_expected:
        raise RuntimeError(f"Expected {gift_rows_expected} gift rows, wrote {gift_rows_written}")

    print("[+] Final validation")
    print(f"    rows written: {total_rows_written}")
    print("    per-user row counts:")
    for pair_type in [PAIR_PI_NO_GC, PAIR_PI_GC, PAIR_NO_PI_GC]:
        print(f"      {pair_type_label(pair_type)}: {per_user_row_pair_counts[pair_type]} rows/user")
    print("    aggregate row counts:")
    for pair_type in [PAIR_PI_NO_GC, PAIR_PI_GC, PAIR_NO_PI_GC]:
        print(f"      {pair_type_label(pair_type)}: {written_pair_counts[pair_type]} rows total")
    print(f"    gift-card rows written: {gift_rows_written}")
    print(f"    no-gift rows written: {total_rows_written - gift_rows_written}")

    return total_rows_written

def validate_proportions(pi_no_gc: float, pi_gc: float, no_pi_gc: float) -> None:
    values = {
        "--pi-no-gc": pi_no_gc,
        "--pi-gc": pi_gc,
        "--no-pi-gc": no_pi_gc,
    }

    for name, value in values.items():
        if not (0.0 <= value <= 1.0):
            raise SystemExit(f"{name} must be between 0 and 1")

    total = pi_no_gc + pi_gc + no_pi_gc
    if not abs(total - 1.0) <= 1e-9:
        raise SystemExit(
            "--pi-no-gc, --pi-gc, and --no-pi-gc must add to 1.0 "
            f"(got {total:.12f})"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-users", type=int, default=100)
    parser.add_argument("--num-candidates", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default="gpt-5-mini")
    parser.add_argument("--format", choices=["jsonl", "csv"], default="jsonl")
    parser.add_argument(
        "--pi-no-gc",
        type=float,
        default=0.34,
        help="Per-user proportion of output rows that should be (P_i, no gift card).",
    )
    parser.add_argument(
        "--pi-gc",
        type=float,
        default=0.33,
        help="Per-user proportion of output rows that should be (P_i, gift card).",
    )
    parser.add_argument(
        "--no-pi-gc",
        type=float,
        default=0.33,
        help="Per-user proportion of output rows that should be (P*_i, gift card).",
    )
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    if args.num_users < 1:
        raise SystemExit("--num-users must be at least 1")
    if args.num_candidates < 1:
        raise SystemExit("--num-candidates must be at least 1")

    validate_proportions(
        pi_no_gc=args.pi_no_gc,
        pi_gc=args.pi_gc,
        no_pi_gc=args.no_pi_gc,
    )

    api_key = get_api_key()
    client = OpenAI(api_key=api_key)

    rng = random.Random(args.seed)
    fake = Faker()
    fake.seed_instance(args.seed)

    output = args.output
    if output is None:
        output = (
            "privacy_sensitive_interactions_api.jsonl"
            if args.format == "jsonl"
            else "privacy_sensitive_interactions_api.csv"
        )

    init_output_file(output, args.format)

    rows_written = generate_dataset(
        num_users=args.num_users,
        model=args.model,
        num_candidates=args.num_candidates,
        rng=rng,
        client=client,
        output=output,
        fmt=args.format,
        pi_no_gc=args.pi_no_gc,
        pi_gc=args.pi_gc,
        no_pi_gc=args.no_pi_gc,
        fake=fake,
    )

    print(f"Wrote {rows_written} rows to {output}")


if __name__ == "__main__":
    main()
