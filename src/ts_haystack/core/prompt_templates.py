# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors
# SPDX-License-Identifier: MIT

"""
Prompt template bank for TS-Haystack.

This module manages diverse natural language templates for questions and answers
to prevent model overfitting on specific phrasings.

Template diversity is inspired by SensorLLM's approach of using 30+ variants per
question type to ensure models learn task semantics rather than surface patterns.
"""

import json
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class TemplateVariant:
    """A single question/answer template with placeholders."""

    question: str
    answer: str


class PromptTemplateBank:
    """
    Manages natural language templates per task to prevent phrasing overfitting.

    Design Principles:
    - Separation of concerns: NLP phrasing is decoupled from task logic
    - Data-driven: Templates can be extended without code changes
    - Grammatically correct: Auto-handles singular/plural, yes/no variants
    - Reproducible: Uses same RNG chain for deterministic template selection
    - High diversity: 20+ templates per task to prevent surface pattern learning

    Usage:
        >>> bank = PromptTemplateBank()
        >>> rng = np.random.default_rng(42)
        >>> q, a = bank.sample("counting", rng, activity="walking", count=3)
        >>> print(q)  # "How many walking bouts occurred in this recording?"
        >>> print(a)  # "There are 3 walking bouts."
    """

    # Templates use {placeholder} syntax for variable substitution
    # Each task has 20+ variants covering: interrogative, imperative, request, conversational forms
    TEMPLATES: Dict[str, List[TemplateVariant]] = {
        # =====================================================================
        # Task 1: Existence - "Is there {activity} in this recording?"
        # =====================================================================
        "existence": [
            # --- Interrogative forms (Is/Does/Can/Did) ---
            TemplateVariant(
                question="Is there any {activity} in this recording?",
                answer="{yes_no}.",
            ),
            TemplateVariant(
                question="Does this recording contain {activity}?",
                answer="{yes_no}, it {does_doesnt}.",
            ),
            TemplateVariant(
                question="Can you detect {activity} in the sensor data?",
                answer="{yes_no}.",
            ),
            TemplateVariant(
                question="Is {activity} present in this accelerometer data?",
                answer="{yes_no}, {activity} is {present_absent}.",
            ),
            TemplateVariant(
                question="Did the person perform {activity} during this period?",
                answer="{yes_no}.",
            ),
            TemplateVariant(
                question="Is there evidence of {activity} in this data?",
                answer="{yes_no}.",
            ),
            TemplateVariant(
                question="Does the accelerometer data show any {activity}?",
                answer="{yes_no}, it {does_doesnt}.",
            ),
            TemplateVariant(
                question="Can {activity} be identified in this recording?",
                answer="{yes_no}.",
            ),
            TemplateVariant(
                question="Is {activity} detectable in the motion data?",
                answer="{yes_no}, {activity} is {present_absent}.",
            ),
            TemplateVariant(
                question="Did {activity} occur at any point in this recording?",
                answer="{yes_no}.",
            ),
            TemplateVariant(
                question="Was {activity} performed during this time window?",
                answer="{yes_no}.",
            ),
            TemplateVariant(
                question="Does this sensor data include {activity}?",
                answer="{yes_no}, it {does_doesnt}.",
            ),
            # --- Imperative forms (Determine/Check/Identify) ---
            TemplateVariant(
                question="Determine if {activity} is present in this recording.",
                answer="{yes_no}, {activity} is {present_absent}.",
            ),
            TemplateVariant(
                question="Check whether this recording contains {activity}.",
                answer="{yes_no}, it {does_doesnt}.",
            ),
            TemplateVariant(
                question="Identify if there is any {activity} in the data.",
                answer="{yes_no}.",
            ),
            TemplateVariant(
                question="Verify whether {activity} occurs in this recording.",
                answer="{yes_no}, it {does_doesnt}.",
            ),
            TemplateVariant(
                question="Assess if the sensor data contains {activity}.",
                answer="{yes_no}.",
            ),
            # --- Request forms (Please/I need/Could you) ---
            TemplateVariant(
                question="Please determine if {activity} occurs in this data.",
                answer="{yes_no}, {activity} is {present_absent}.",
            ),
            TemplateVariant(
                question="I need to know if {activity} is present. Can you tell?",
                answer="{yes_no}, {activity} is {present_absent}.",
            ),
            TemplateVariant(
                question="Could you check if there's any {activity} in this recording?",
                answer="{yes_no}.",
            ),
            TemplateVariant(
                question="Please verify whether {activity} appears in the data.",
                answer="{yes_no}, it {does_doesnt}.",
            ),
            TemplateVariant(
                question="Would you determine if {activity} is detectable here?",
                answer="{yes_no}.",
            ),
            # --- Conversational forms ---
            TemplateVariant(
                question="Looking at this data, is there any sign of {activity}?",
                answer="{yes_no}.",
            ),
            TemplateVariant(
                question="Based on the sensor readings, did {activity} happen?",
                answer="{yes_no}.",
            ),
            TemplateVariant(
                question="From the accelerometer data, can you tell if {activity} occurred?",
                answer="{yes_no}, {activity} is {present_absent}.",
            ),
            TemplateVariant(
                question="Given this recording, is {activity} among the activities?",
                answer="{yes_no}.",
            ),
            TemplateVariant(
                question="Examining the motion data, is {activity} visible?",
                answer="{yes_no}, it {does_doesnt} appear.",
            ),
        ],
        # =====================================================================
        # Task 2: Localization - "When did {activity} occur?"
        # =====================================================================
        "localization": [
            # --- Interrogative forms ---
            TemplateVariant(
                question="When did the {activity} bout occur?",
                answer="The {activity} bout occurred from {start} to {end}.",
            ),
            TemplateVariant(
                question="At what time was {activity} detected?",
                answer="{activity} was detected between {start} and {end}.",
            ),
            TemplateVariant(
                question="During which time period did {activity} take place?",
                answer="{activity} took place from {start} to {end}.",
            ),
            TemplateVariant(
                question="What is the time range of the {activity} bout?",
                answer="The {activity} bout spans from {start} to {end}.",
            ),
            TemplateVariant(
                question="When was {activity} recorded in this data?",
                answer="{activity} was recorded from {start} to {end}.",
            ),
            TemplateVariant(
                question="At what point did {activity} happen?",
                answer="{activity} happened from {start} to {end}.",
            ),
            TemplateVariant(
                question="What time did {activity} start and end?",
                answer="{activity} started at {start} and ended at {end}.",
            ),
            TemplateVariant(
                question="During what interval was {activity} observed?",
                answer="{activity} was observed from {start} to {end}.",
            ),
            TemplateVariant(
                question="When exactly did the {activity} episode occur?",
                answer="The {activity} episode occurred from {start} to {end}.",
            ),
            TemplateVariant(
                question="What are the timestamps for the {activity} bout?",
                answer="The {activity} bout: {start} to {end}.",
            ),
            # --- Imperative forms ---
            TemplateVariant(
                question="Identify when {activity} happened.",
                answer="From {start} to {end}.",
            ),
            TemplateVariant(
                question="Locate the {activity} bout in this recording.",
                answer="The {activity} bout is located from {start} to {end}.",
            ),
            TemplateVariant(
                question="Find the time period when {activity} occurred.",
                answer="{activity} occurred from {start} to {end}.",
            ),
            TemplateVariant(
                question="Pinpoint when the {activity} event took place.",
                answer="The {activity} event took place from {start} to {end}.",
            ),
            TemplateVariant(
                question="Determine the timing of the {activity} bout.",
                answer="The {activity} bout timing: {start} to {end}.",
            ),
            TemplateVariant(
                question="Specify when {activity} was detected.",
                answer="{activity} was detected from {start} to {end}.",
            ),
            # --- Request forms ---
            TemplateVariant(
                question="Please identify when {activity} occurred.",
                answer="{activity} occurred from {start} to {end}.",
            ),
            TemplateVariant(
                question="Could you locate the {activity} bout in the data?",
                answer="The {activity} bout is from {start} to {end}.",
            ),
            TemplateVariant(
                question="I need to know when {activity} happened.",
                answer="{activity} happened from {start} to {end}.",
            ),
            TemplateVariant(
                question="Would you find the time range for {activity}?",
                answer="The time range for {activity}: {start} to {end}.",
            ),
            TemplateVariant(
                question="Please tell me the timestamps for {activity}.",
                answer="{activity}: {start} to {end}.",
            ),
            # --- Conversational forms ---
            TemplateVariant(
                question="Looking at this recording, when did {activity} occur?",
                answer="{activity} occurred from {start} to {end}.",
            ),
            TemplateVariant(
                question="In this sensor data, at what time was {activity} detected?",
                answer="{activity} was detected from {start} to {end}.",
            ),
            TemplateVariant(
                question="Based on the accelerometer readings, when did {activity} happen?",
                answer="{activity} happened from {start} to {end}.",
            ),
            TemplateVariant(
                question="From the motion data, identify the {activity} time window.",
                answer="The {activity} time window: {start} to {end}.",
            ),
        ],
        # =====================================================================
        # Task 3: Counting - "How many {activity} bouts occurred?"
        # =====================================================================
        "counting": [
            # --- Interrogative forms ---
            TemplateVariant(
                question="How many {activity} bouts occurred in this recording?",
                answer="There {is_are} {count} {activity} {bout_bouts}.",
            ),
            TemplateVariant(
                question="How many times did {activity} occur?",
                answer="{activity} occurred {count} {time_times}.",
            ),
            TemplateVariant(
                question="What is the total count of {activity} bouts?",
                answer="The total count is {count}.",
            ),
            TemplateVariant(
                question="How many {activity} episodes are in this data?",
                answer="There {is_are} {count} {activity} {episode_episodes}.",
            ),
            TemplateVariant(
                question="How many instances of {activity} can be found?",
                answer="{count} {instance_instances} of {activity} can be found.",
            ),
            TemplateVariant(
                question="What is the number of {activity} bouts in this recording?",
                answer="The number is {count}.",
            ),
            TemplateVariant(
                question="How often did {activity} occur in this data?",
                answer="{activity} occurred {count} {time_times}.",
            ),
            TemplateVariant(
                question="How many separate {activity} events are there?",
                answer="There {is_are} {count} separate {activity} {event_events}.",
            ),
            TemplateVariant(
                question="What is the frequency of {activity} bouts?",
                answer="There {is_are} {count} {activity} {bout_bouts}.",
            ),
            TemplateVariant(
                question="How many distinct {activity} periods are present?",
                answer="{count} distinct {activity} {period_periods} {is_are} present.",
            ),
            # --- Imperative forms ---
            TemplateVariant(
                question="Count the number of {activity} episodes.",
                answer="{count}.",
            ),
            TemplateVariant(
                question="Determine how many {activity} bouts occurred.",
                answer="{count} {activity} {bout_bouts} occurred.",
            ),
            TemplateVariant(
                question="Calculate the total {activity} occurrences.",
                answer="Total: {count}.",
            ),
            TemplateVariant(
                question="Find the count of {activity} events in the data.",
                answer="Count: {count}.",
            ),
            TemplateVariant(
                question="Tally the {activity} bouts in this recording.",
                answer="Tally: {count} {activity} {bout_bouts}.",
            ),
            TemplateVariant(
                question="Enumerate the {activity} episodes.",
                answer="There {is_are} {count} {episode_episodes}.",
            ),
            # --- Request forms ---
            TemplateVariant(
                question="Please count the {activity} bouts.",
                answer="There {is_are} {count} {activity} {bout_bouts}.",
            ),
            TemplateVariant(
                question="Could you tell me how many times {activity} occurred?",
                answer="{activity} occurred {count} {time_times}.",
            ),
            TemplateVariant(
                question="I need the count of {activity} events.",
                answer="The count is {count}.",
            ),
            TemplateVariant(
                question="Would you determine the number of {activity} bouts?",
                answer="The number is {count}.",
            ),
            TemplateVariant(
                question="Please identify how many {activity} episodes there are.",
                answer="There {is_are} {count} {episode_episodes}.",
            ),
            # --- Conversational forms ---
            TemplateVariant(
                question="Looking at this data, how many {activity} bouts are there?",
                answer="There {is_are} {count} {activity} {bout_bouts}.",
            ),
            TemplateVariant(
                question="In this recording, how often does {activity} appear?",
                answer="{activity} appears {count} {time_times}.",
            ),
            TemplateVariant(
                question="Based on the sensor readings, how many {activity} events occurred?",
                answer="{count} {activity} {event_events} occurred.",
            ),
            TemplateVariant(
                question="From the accelerometer data, count the {activity} occurrences.",
                answer="There {is_are} {count} {occurrence_occurrences}.",
            ),
        ],
        # =====================================================================
        # Task 4: Ordering - "Did {activity_a} occur before {activity_b}?"
        # =====================================================================
        "ordering": [
            # --- Boolean question forms (Did/Was/Is) ---
            TemplateVariant(
                question="Did {activity_a} occur before {activity_b}?",
                answer="{yes_no}.",
            ),
            TemplateVariant(
                question="Was {activity_a} performed before {activity_b}?",
                answer="{yes_no}, {activity_a} {came_before_after} {activity_b}.",
            ),
            TemplateVariant(
                question="Did {activity_a} happen prior to {activity_b}?",
                answer="{yes_no}.",
            ),
            TemplateVariant(
                question="Is {activity_a} earlier than {activity_b} in this recording?",
                answer="{yes_no}.",
            ),
            TemplateVariant(
                question="Did {activity_a} precede {activity_b}?",
                answer="{yes_no}, {activity_a} {came_before_after} {activity_b}.",
            ),
            TemplateVariant(
                question="Was {activity_a} detected before {activity_b}?",
                answer="{yes_no}.",
            ),
            TemplateVariant(
                question="Did {activity_b} come after {activity_a}?",
                answer="{yes_no}.",
            ),
            TemplateVariant(
                question="Is it true that {activity_a} occurred before {activity_b}?",
                answer="{yes_no}.",
            ),
            TemplateVariant(
                question="Was {activity_b} subsequent to {activity_a}?",
                answer="{yes_no}.",
            ),
            # --- Choice question forms (Which) ---
            TemplateVariant(
                question="Which occurred first, {activity_a} or {activity_b}?",
                answer="{first_activity}.",
            ),
            TemplateVariant(
                question="Which activity happened earlier: {activity_a} or {activity_b}?",
                answer="{first_activity} happened earlier.",
            ),
            TemplateVariant(
                question="Between {activity_a} and {activity_b}, which came first?",
                answer="{first_activity} came first.",
            ),
            TemplateVariant(
                question="Which of the two activities was detected first: {activity_a} or {activity_b}?",
                answer="{first_activity}.",
            ),
            TemplateVariant(
                question="Of {activity_a} and {activity_b}, which preceded the other?",
                answer="{first_activity} preceded {second_activity}.",
            ),
            # --- Order description forms ---
            TemplateVariant(
                question="In what order did {activity_a} and {activity_b} happen?",
                answer="{first_activity} occurred first, followed by {second_activity}.",
            ),
            TemplateVariant(
                question="What is the temporal order of {activity_a} and {activity_b}?",
                answer="{first_activity} then {second_activity}.",
            ),
            TemplateVariant(
                question="Describe the sequence of {activity_a} and {activity_b}.",
                answer="{first_activity} occurred before {second_activity}.",
            ),
            # --- Imperative forms ---
            TemplateVariant(
                question="Determine if {activity_a} occurred before {activity_b}.",
                answer="{yes_no}, {activity_a} {came_before_after} {activity_b}.",
            ),
            TemplateVariant(
                question="Identify which happened first: {activity_a} or {activity_b}.",
                answer="{first_activity}.",
            ),
            TemplateVariant(
                question="Establish the order of {activity_a} and {activity_b}.",
                answer="{first_activity} occurred first, then {second_activity}.",
            ),
            # --- Request forms ---
            TemplateVariant(
                question="Please tell me if {activity_a} came before {activity_b}.",
                answer="{yes_no}.",
            ),
            TemplateVariant(
                question="Could you determine which occurred first: {activity_a} or {activity_b}?",
                answer="{first_activity} occurred first.",
            ),
            TemplateVariant(
                question="I need to know the order of {activity_a} and {activity_b}.",
                answer="{first_activity} came first, followed by {second_activity}.",
            ),
            # --- Conversational forms ---
            TemplateVariant(
                question="Looking at the data, did {activity_a} happen before {activity_b}?",
                answer="{yes_no}.",
            ),
            TemplateVariant(
                question="In this recording, which came first: {activity_a} or {activity_b}?",
                answer="{first_activity}.",
            ),
        ],
        # =====================================================================
        # Task 5: State Query - "What was the activity level when {event} occurred?"
        # =====================================================================
        "state_query": [
            # --- Interrogative forms ---
            TemplateVariant(
                question="What was the overall activity level when the {needle_activity} spike occurred?",
                answer="The overall activity level was {global_state}.",
            ),
            TemplateVariant(
                question="During the {needle_activity} event, what was the general activity regime?",
                answer="{global_state}.",
            ),
            TemplateVariant(
                question="What activity state was the person in when {needle_activity} happened?",
                answer="The person was in a {global_state} state.",
            ),
            TemplateVariant(
                question="What was the background activity when {needle_activity} occurred?",
                answer="The background activity was {global_state}.",
            ),
            TemplateVariant(
                question="What was the dominant activity during the {needle_activity} event?",
                answer="The dominant activity was {global_state}.",
            ),
            TemplateVariant(
                question="At the time of {needle_activity}, what was the overall activity?",
                answer="The overall activity was {global_state}.",
            ),
            TemplateVariant(
                question="What activity regime surrounded the {needle_activity} spike?",
                answer="The surrounding regime was {global_state}.",
            ),
            TemplateVariant(
                question="During {needle_activity}, what was the person generally doing?",
                answer="The person was generally {global_state}.",
            ),
            TemplateVariant(
                question="What was the activity context when {needle_activity} was detected?",
                answer="The activity context was {global_state}.",
            ),
            TemplateVariant(
                question="What global state was present during the {needle_activity} occurrence?",
                answer="The global state was {global_state}.",
            ),
            # --- Imperative forms ---
            TemplateVariant(
                question="Identify the activity level when {needle_activity} occurred.",
                answer="The activity level was {global_state}.",
            ),
            TemplateVariant(
                question="Determine the background activity during {needle_activity}.",
                answer="The background activity was {global_state}.",
            ),
            TemplateVariant(
                question="Find the overall activity state at the time of {needle_activity}.",
                answer="The overall state was {global_state}.",
            ),
            TemplateVariant(
                question="Describe the activity regime when {needle_activity} happened.",
                answer="The regime was {global_state}.",
            ),
            # --- Request forms ---
            TemplateVariant(
                question="Please identify what the person was doing when {needle_activity} occurred.",
                answer="The person was {global_state}.",
            ),
            TemplateVariant(
                question="Could you tell me the activity state during {needle_activity}?",
                answer="The activity state was {global_state}.",
            ),
            TemplateVariant(
                question="I need to know the background activity when {needle_activity} happened.",
                answer="The background activity was {global_state}.",
            ),
            TemplateVariant(
                question="What was going on overall when the {needle_activity} event occurred?",
                answer="Overall, the activity was {global_state}.",
            ),
            # --- Conversational forms ---
            TemplateVariant(
                question="Looking at the broader context, what activity was happening during {needle_activity}?",
                answer="The broader activity was {global_state}.",
            ),
            TemplateVariant(
                question="When {needle_activity} was detected, what was the general activity pattern?",
                answer="The general pattern was {global_state}.",
            ),
            TemplateVariant(
                question="In the context of this recording, what activity surrounded {needle_activity}?",
                answer="The surrounding activity was {global_state}.",
            ),
        ],
        # =====================================================================
        # Task 6: Antecedent - "What activity occurred before {target}?"
        # =====================================================================
        "antecedent": [
            # --- Interrogative forms ---
            TemplateVariant(
                question="What activity occurred immediately before the {target_activity} bout?",
                answer="{antecedent_activity}.",
            ),
            TemplateVariant(
                question="Which activity preceded {target_activity}?",
                answer="{antecedent_activity} preceded {target_activity}.",
            ),
            TemplateVariant(
                question="What was the person doing right before {target_activity}?",
                answer="The person was {antecedent_activity}.",
            ),
            TemplateVariant(
                question="What activity came just before {target_activity}?",
                answer="{antecedent_activity}.",
            ),
            TemplateVariant(
                question="What happened immediately prior to {target_activity}?",
                answer="{antecedent_activity} happened prior to {target_activity}.",
            ),
            TemplateVariant(
                question="Which activity was detected right before {target_activity}?",
                answer="{antecedent_activity} was detected before {target_activity}.",
            ),
            TemplateVariant(
                question="What was the preceding activity before {target_activity}?",
                answer="The preceding activity was {antecedent_activity}.",
            ),
            TemplateVariant(
                question="What activity directly preceded the {target_activity} bout?",
                answer="{antecedent_activity} directly preceded it.",
            ),
            TemplateVariant(
                question="Just before {target_activity}, what was happening?",
                answer="{antecedent_activity} was happening.",
            ),
            TemplateVariant(
                question="What was the activity immediately before {target_activity} started?",
                answer="{antecedent_activity}.",
            ),
            # --- Imperative forms ---
            TemplateVariant(
                question="Identify the activity that occurred before {target_activity}.",
                answer="{antecedent_activity}.",
            ),
            TemplateVariant(
                question="Find what preceded {target_activity} in this recording.",
                answer="{antecedent_activity} preceded {target_activity}.",
            ),
            TemplateVariant(
                question="Determine the antecedent activity to {target_activity}.",
                answer="The antecedent activity was {antecedent_activity}.",
            ),
            TemplateVariant(
                question="Locate the activity that came before {target_activity}.",
                answer="{antecedent_activity}.",
            ),
            # --- Request forms ---
            TemplateVariant(
                question="Please tell me what happened before {target_activity}.",
                answer="{antecedent_activity} happened before {target_activity}.",
            ),
            TemplateVariant(
                question="Could you identify the preceding activity to {target_activity}?",
                answer="The preceding activity was {antecedent_activity}.",
            ),
            TemplateVariant(
                question="I need to know what activity came before {target_activity}.",
                answer="{antecedent_activity} came before {target_activity}.",
            ),
            TemplateVariant(
                question="What activity would you say preceded {target_activity}?",
                answer="{antecedent_activity} preceded {target_activity}.",
            ),
            # --- Conversational forms ---
            TemplateVariant(
                question="Looking at the sequence, what came before {target_activity}?",
                answer="{antecedent_activity} came before {target_activity}.",
            ),
            TemplateVariant(
                question="In this data, what activity led up to {target_activity}?",
                answer="{antecedent_activity} led up to {target_activity}.",
            ),
            TemplateVariant(
                question="Before {target_activity} began, what was the person doing?",
                answer="The person was {antecedent_activity}.",
            ),
        ],
        # =====================================================================
        # Task 7a: Comparison WITH activity - "What was the {extremum} {activity} bout?"
        # Used when polarity="with" - finding longest/shortest activity BOUTS
        # =====================================================================
        "comparison_with": [
            # --- Interrogative forms ---
            TemplateVariant(
                question="What is the {extremum} {activity} bout in this recording?",
                answer="The {extremum} bout is from {start} to {end} ({duration}).",
            ),
            TemplateVariant(
                question="Which {activity} bout was the {extremum}?",
                answer="The bout from {start} to {end} was {extremum} ({duration}).",
            ),
            TemplateVariant(
                question="What was the time range of the {extremum} {activity} episode?",
                answer="The {extremum} episode: {start} to {end}.",
            ),
            TemplateVariant(
                question="When did the {extremum} {activity} bout occur?",
                answer="The {extremum} bout occurred from {start} to {end}.",
            ),
            TemplateVariant(
                question="What are the timestamps of the {extremum} {activity} period?",
                answer="Timestamps: {start} to {end} ({duration}).",
            ),
            TemplateVariant(
                question="Which {activity} event had the {extremum} duration?",
                answer="The event from {start} to {end} ({duration}).",
            ),
            TemplateVariant(
                question="What was the {extremum} {activity} period in the data?",
                answer="The {extremum} period was from {start} to {end}.",
            ),
            TemplateVariant(
                question="When was the {extremum} stretch of {activity}?",
                answer="The {extremum} stretch of {activity} was from {start} to {end}.",
            ),
            # --- Imperative forms ---
            TemplateVariant(
                question="Identify the {extremum} {activity} bout.",
                answer="From {start} to {end} ({duration}).",
            ),
            TemplateVariant(
                question="Find the {extremum} {activity} bout in this recording.",
                answer="The {extremum} bout: {start} to {end}.",
            ),
            TemplateVariant(
                question="Locate the {extremum} {activity} episode in this data.",
                answer="The {extremum} episode: {start} to {end} ({duration}).",
            ),
            TemplateVariant(
                question="Determine the {extremum} stretch of {activity}.",
                answer="The {extremum} stretch: {start} to {end}.",
            ),
            TemplateVariant(
                question="Pinpoint the {extremum} {activity} bout.",
                answer="{start} to {end} ({duration}).",
            ),
            # --- Request forms ---
            TemplateVariant(
                question="Please identify the {extremum} {activity} bout.",
                answer="The {extremum} bout: {start} to {end} ({duration}).",
            ),
            TemplateVariant(
                question="Could you find the {extremum} {activity} period?",
                answer="The {extremum} period: {start} to {end}.",
            ),
            TemplateVariant(
                question="I need to know which {activity} bout was {extremum}.",
                answer="The bout from {start} to {end} was {extremum} ({duration}).",
            ),
            TemplateVariant(
                question="Would you locate the {extremum} {activity} episode?",
                answer="The {extremum} episode: {start} to {end}.",
            ),
            # --- Conversational forms ---
            TemplateVariant(
                question="Looking at all {activity} bouts, which was {extremum}?",
                answer="The bout from {start} to {end} was {extremum} ({duration}).",
            ),
            TemplateVariant(
                question="Among the {activity} episodes, which one had the {extremum} duration?",
                answer="The episode from {start} to {end} ({duration}).",
            ),
            TemplateVariant(
                question="In this recording, when was the {extremum} {activity} stretch?",
                answer="The {extremum} stretch: {start} to {end}.",
            ),
        ],
        # =====================================================================
        # Task 7b: Comparison WITHOUT activity - "What was the {extremum} gap without {activity}?"
        # Used when polarity="without" - finding longest/shortest GAPS between bouts
        # =====================================================================
        "comparison_without": [
            # --- Interrogative forms ---
            TemplateVariant(
                question="What was the {extremum} period without {activity}?",
                answer="The {extremum} period without {activity} was from {start} to {end} ({duration}).",
            ),
            TemplateVariant(
                question="When was the {extremum} stretch without {activity}?",
                answer="The {extremum} stretch without {activity} was from {start} to {end}.",
            ),
            TemplateVariant(
                question="What is the {extremum} gap between {activity} bouts?",
                answer="The {extremum} gap is from {start} to {end} ({duration}).",
            ),
            TemplateVariant(
                question="Which period without {activity} was the {extremum}?",
                answer="The period from {start} to {end} was {extremum} ({duration}).",
            ),
            TemplateVariant(
                question="What was the time range of the {extremum} interval without {activity}?",
                answer="The {extremum} interval: {start} to {end}.",
            ),
            TemplateVariant(
                question="When did the {extremum} gap without {activity} occur?",
                answer="The {extremum} gap occurred from {start} to {end}.",
            ),
            TemplateVariant(
                question="What are the timestamps of the {extremum} period lacking {activity}?",
                answer="Timestamps: {start} to {end} ({duration}).",
            ),
            TemplateVariant(
                question="Which interval between {activity} bouts had the {extremum} duration?",
                answer="The interval from {start} to {end} ({duration}).",
            ),
            # --- Imperative forms ---
            TemplateVariant(
                question="Identify the {extremum} gap without {activity}.",
                answer="From {start} to {end} ({duration}).",
            ),
            TemplateVariant(
                question="Find the {extremum} period without {activity}.",
                answer="The {extremum} period: {start} to {end}.",
            ),
            TemplateVariant(
                question="Locate the {extremum} interval lacking {activity} in this data.",
                answer="The {extremum} interval: {start} to {end} ({duration}).",
            ),
            TemplateVariant(
                question="Determine the {extremum} stretch without {activity}.",
                answer="The {extremum} stretch: {start} to {end}.",
            ),
            TemplateVariant(
                question="Pinpoint the {extremum} gap between {activity} bouts.",
                answer="{start} to {end} ({duration}).",
            ),
            # --- Request forms ---
            TemplateVariant(
                question="Please identify the {extremum} period without {activity}.",
                answer="The {extremum} period: {start} to {end} ({duration}).",
            ),
            TemplateVariant(
                question="Could you find the {extremum} gap without {activity}?",
                answer="The {extremum} gap: {start} to {end}.",
            ),
            TemplateVariant(
                question="I need to know which interval without {activity} was {extremum}.",
                answer="The interval from {start} to {end} was {extremum} ({duration}).",
            ),
            TemplateVariant(
                question="Would you locate the {extremum} stretch without {activity}?",
                answer="The {extremum} stretch: {start} to {end}.",
            ),
            # --- Conversational forms ---
            TemplateVariant(
                question="Looking at all gaps between {activity} bouts, which was {extremum}?",
                answer="The gap from {start} to {end} was {extremum} ({duration}).",
            ),
            TemplateVariant(
                question="Among the periods without {activity}, which one was {extremum}?",
                answer="The period from {start} to {end} ({duration}).",
            ),
            TemplateVariant(
                question="In this recording, when was the {extremum} interval without {activity}?",
                answer="The {extremum} interval: {start} to {end}.",
            ),
        ],
        # =====================================================================
        # Task 8: Multi-Hop - "When did the Nth {target} occur {before/after} {anchor}?"
        # =====================================================================
        "multi_hop": [
            # --- Interrogative forms ---
            TemplateVariant(
                question="When did the {ordinal} {target_activity} bout occur {direction} the {anchor_activity}?",
                answer="The {ordinal} {target_activity} bout {direction} {anchor_activity} occurred from {start} to {end}.",
            ),
            TemplateVariant(
                question="What is the timing of the {ordinal} {target_activity} bout {direction} {anchor_activity}?",
                answer="The timing is {start} to {end}.",
            ),
            TemplateVariant(
                question="When was the {ordinal} {target_activity} episode {direction} {anchor_activity}?",
                answer="The {ordinal} episode was from {start} to {end}.",
            ),
            TemplateVariant(
                question="At what time did the {ordinal} {target_activity} bout happen {direction} {anchor_activity}?",
                answer="It happened from {start} to {end}.",
            ),
            TemplateVariant(
                question="What are the timestamps for the {ordinal} {target_activity} activity {direction} {anchor_activity}?",
                answer="Timestamps: {start} to {end}.",
            ),
            TemplateVariant(
                question="When did the {ordinal} occurrence of {target_activity} {direction} {anchor_activity} take place?",
                answer="It took place from {start} to {end}.",
            ),
            TemplateVariant(
                question="Where in the recording is the {ordinal} {target_activity} activity {direction} {anchor_activity}?",
                answer="It is located from {start} to {end}.",
            ),
            # --- Imperative forms ---
            TemplateVariant(
                question="Identify the {ordinal} {target_activity} bout {direction} {anchor_activity}.",
                answer="From {start} to {end}.",
            ),
            TemplateVariant(
                question="After locating {anchor_activity}, find the {ordinal} {target_activity} activity {direction} it.",
                answer="The {ordinal} {target_activity}: {start} to {end}.",
            ),
            TemplateVariant(
                question="Locate the {ordinal} {target_activity} bout {direction} the {anchor_activity} event.",
                answer="Located from {start} to {end}.",
            ),
            TemplateVariant(
                question="Find the {ordinal} instance of {target_activity} {direction} {anchor_activity}.",
                answer="The {ordinal} instance: {start} to {end}.",
            ),
            TemplateVariant(
                question="Determine when the {ordinal} {target_activity} activity occurred {direction} {anchor_activity}.",
                answer="It occurred from {start} to {end}.",
            ),
            TemplateVariant(
                question="Pinpoint the {ordinal} {target_activity} bout {direction} {anchor_activity}.",
                answer="{start} to {end}.",
            ),
            # --- Request forms ---
            TemplateVariant(
                question="After locating {anchor_activity}, when was the {ordinal} {target_activity} activity bout {direction} it?",
                answer="It occurred from {start} to {end}.",
            ),
            TemplateVariant(
                question="Please find the {ordinal} {target_activity} activity {direction} {anchor_activity}.",
                answer="The {ordinal} {target_activity}: {start} to {end}.",
            ),
            TemplateVariant(
                question="Could you identify the {ordinal} {target_activity} bout {direction} {anchor_activity}?",
                answer="The bout is from {start} to {end}.",
            ),
            TemplateVariant(
                question="I need the timing of the {ordinal} {target_activity} activity {direction} {anchor_activity}.",
                answer="The timing: {start} to {end}.",
            ),
            TemplateVariant(
                question="Would you locate the {ordinal} occurrence of {target_activity} {direction} {anchor_activity}?",
                answer="The occurrence: {start} to {end}.",
            ),
            # --- Conversational forms ---
            TemplateVariant(
                question="First find {anchor_activity}, then tell me when the {ordinal} {target_activity} activity occurred {direction} it.",
                answer="The {ordinal} {target_activity} {direction} {anchor_activity}: {start} to {end}.",
            ),
            TemplateVariant(
                question="Using {anchor_activity} as a reference, when was the {ordinal} {target_activity} bout {direction} it?",
                answer="The {ordinal} {target_activity} was from {start} to {end}.",
            ),
            TemplateVariant(
                question="Relative to {anchor_activity}, identify the {ordinal} {target_activity} activity {direction} it.",
                answer="The {ordinal} {target_activity}: {start} to {end}.",
            ),
            TemplateVariant(
                question="Looking at the sequence around {anchor_activity}, when did the {ordinal} {target_activity} bout occur {direction} it?",
                answer="It occurred from {start} to {end}.",
            ),
        ],
        # =====================================================================
        # Task 9a: Anomaly Detection (Positive) - Anomaly IS present
        # Tests contextual reasoning: detecting cross-regime activity insertions
        # Requires: anomaly_activity, background_regime
        # =====================================================================
        "anomaly_detection_positive": [
            TemplateVariant(
                question="Is there an anomaly in this recording?",
                answer="Yes, there is anomalous {anomaly_activity} activity in the {background_regime} background.",
            ),
            TemplateVariant(
                question="Is there any anomalous activity in this accelerometer data?",
                answer="Yes, {anomaly_activity} is anomalous in the {background_regime} context.",
            ),
            TemplateVariant(
                question="Does this recording contain any unusual activity patterns?",
                answer="Yes, there is unusual {anomaly_activity} activity within the {background_regime} background.",
            ),
            TemplateVariant(
                question="Can you detect any anomalies in this sensor data?",
                answer="Yes, I detect anomalous {anomaly_activity} in the otherwise {background_regime} recording.",
            ),
            TemplateVariant(
                question="Is there anything out of the ordinary in this recording?",
                answer="Yes, {anomaly_activity} activity is out of the ordinary for the {background_regime} background.",
            ),
            TemplateVariant(
                question="Determine if there is an anomaly in this data.",
                answer="Yes, there is an anomalous {anomaly_activity} bout in the {background_regime} background.",
            ),
            TemplateVariant(
                question="Check for any anomalous patterns in this recording.",
                answer="Yes, {anomaly_activity} is anomalous relative to the {background_regime} context.",
            ),
            TemplateVariant(
                question="Please identify if any activity is anomalous in this recording.",
                answer="Yes, {anomaly_activity} activity is anomalous in the {background_regime} background.",
            ),
            TemplateVariant(
                question="Is there anything unusual that doesn't fit the overall pattern?",
                answer="Yes, {anomaly_activity} does not fit the {background_regime} pattern.",
            ),
            TemplateVariant(
                question="Looking at this data, can you identify any anomalies?",
                answer="Yes, there is anomalous {anomaly_activity} in the {background_regime} background.",
            ),
        ],
        # =====================================================================
        # Task 9b: Anomaly Detection (Negative) - No anomaly present
        # Same-regime insertions are NOT anomalous
        # Requires: background_regime
        # =====================================================================
        "anomaly_detection_negative": [
            TemplateVariant(
                question="Is there an anomaly in this recording?",
                answer="No, the recording shows consistent {background_regime} activity.",
            ),
            TemplateVariant(
                question="Is there any anomalous activity in this accelerometer data?",
                answer="No, all activities are consistent with the {background_regime} pattern.",
            ),
            TemplateVariant(
                question="Does this recording contain any unusual activity patterns?",
                answer="No, the recording shows typical {background_regime} activity throughout.",
            ),
            TemplateVariant(
                question="Can you detect any anomalies in this sensor data?",
                answer="No anomalies detected. The data shows consistent {background_regime} activity.",
            ),
            TemplateVariant(
                question="Is there anything out of the ordinary in this recording?",
                answer="No, everything is consistent with {background_regime} activity.",
            ),
            TemplateVariant(
                question="Determine if there is an anomaly in this data.",
                answer="No, the recording is consistent with {background_regime} activity.",
            ),
            TemplateVariant(
                question="Check for any anomalous patterns in this recording.",
                answer="No anomalous patterns detected. The {background_regime} pattern is consistent.",
            ),
            TemplateVariant(
                question="Please identify if any activity is anomalous in this recording.",
                answer="No, all activities are appropriate for the {background_regime} context.",
            ),
            TemplateVariant(
                question="Is there anything unusual that doesn't fit the overall pattern?",
                answer="No, all activities fit the {background_regime} pattern.",
            ),
            TemplateVariant(
                question="Looking at this data, can you identify any anomalies?",
                answer="No anomalies identified. The data shows {background_regime} activity.",
            ),
        ],
        # =====================================================================
        # Task 10a: Anomaly Localization (Positive) - Anomaly IS present with time range
        # Combines anomaly detection with temporal localization.
        # Requires: anomaly_activity, start, end
        # =====================================================================
        "anomaly_localization_positive": [
            TemplateVariant(
                question="Is there an anomaly in this recording, and if so, when does it occur?",
                answer="Yes, there is anomalous {anomaly_activity} activity from {start} to {end}.",
            ),
            TemplateVariant(
                question="Identify any anomalies and their timing in this data.",
                answer="Yes, {anomaly_activity} is anomalous, occurring from {start} to {end}.",
            ),
            TemplateVariant(
                question="Is there any unusual activity? If yes, specify when it occurred.",
                answer="Yes, unusual {anomaly_activity} activity occurred from {start} to {end}.",
            ),
            TemplateVariant(
                question="Detect and locate any anomalies in this accelerometer data.",
                answer="Anomaly detected: {anomaly_activity} from {start} to {end}.",
            ),
            TemplateVariant(
                question="Does this recording contain anomalies? Provide the time range if so.",
                answer="Yes, anomalous {anomaly_activity} from {start} to {end}.",
            ),
            TemplateVariant(
                question="Find any anomalous patterns and specify when they occur.",
                answer="Anomalous {anomaly_activity} found from {start} to {end}.",
            ),
            TemplateVariant(
                question="Is there anything out of the ordinary? Report the timing.",
                answer="Yes, {anomaly_activity} is out of the ordinary, occurring {start} to {end}.",
            ),
            TemplateVariant(
                question="Check for anomalies and report their temporal location.",
                answer="Anomaly: {anomaly_activity} from {start} to {end}.",
            ),
            TemplateVariant(
                question="Determine if there is an anomaly and when it happened.",
                answer="Yes, there is anomalous {anomaly_activity} from {start} to {end}.",
            ),
            TemplateVariant(
                question="Identify and localize any anomalous activity in this recording.",
                answer="Anomalous {anomaly_activity} localized at {start} to {end}.",
            ),
        ],
        # =====================================================================
        # Task 10b: Anomaly Localization (Negative) - No anomaly present
        # Same-regime insertions are NOT anomalous
        # Requires: background_regime
        # =====================================================================
        "anomaly_localization_negative": [
            TemplateVariant(
                question="Is there an anomaly in this recording, and if so, when does it occur?",
                answer="No, the recording shows consistent {background_regime} activity.",
            ),
            TemplateVariant(
                question="Identify any anomalies and their timing in this data.",
                answer="No anomalies found. The data shows consistent {background_regime} activity.",
            ),
            TemplateVariant(
                question="Is there any unusual activity? If yes, specify when it occurred.",
                answer="No unusual activity. The recording is consistent with {background_regime} patterns.",
            ),
            TemplateVariant(
                question="Detect and locate any anomalies in this accelerometer data.",
                answer="No anomalies detected. Consistent {background_regime} activity throughout.",
            ),
            TemplateVariant(
                question="Does this recording contain anomalies? Provide the time range if so.",
                answer="No, the recording shows consistent {background_regime} activity.",
            ),
            TemplateVariant(
                question="Find any anomalous patterns and specify when they occur.",
                answer="No anomalous patterns found in this {background_regime} recording.",
            ),
            TemplateVariant(
                question="Is there anything out of the ordinary? Report the timing.",
                answer="Nothing out of the ordinary. Consistent {background_regime} activity.",
            ),
            TemplateVariant(
                question="Check for anomalies and report their temporal location.",
                answer="No anomalies to report. The {background_regime} pattern is consistent.",
            ),
            TemplateVariant(
                question="Determine if there is an anomaly and when it happened.",
                answer="No anomaly detected. The recording shows {background_regime} activity.",
            ),
            TemplateVariant(
                question="Identify and localize any anomalous activity in this recording.",
                answer="No anomalous activity found. Consistent {background_regime} throughout.",
            ),
        ],
    }

    # Ordinal mappings for multi-hop
    ORDINALS = {1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth"}

    def __init__(self, custom_templates: Optional[Dict[str, List[TemplateVariant]]] = None):
        """
        Initialize with default templates, optionally extended with custom ones.

        Args:
            custom_templates: Additional templates to merge with defaults
        """
        self.templates = {task: list(variants) for task, variants in self.TEMPLATES.items()}
        if custom_templates:
            for task, variants in custom_templates.items():
                self.templates.setdefault(task, []).extend(variants)

    def sample(
        self,
        task: str,
        rng: np.random.Generator,
        **kwargs,
    ) -> Tuple[str, str]:
        """
        Sample a random template and fill placeholders.

        Args:
            task: Task name (existence, localization, etc.)
            rng: Random generator for reproducible selection
            **kwargs: Placeholder values (activity, count, start, end, etc.)

        Returns:
            (question, answer) tuple with placeholders filled

        Example:
            >>> bank.sample("counting", rng, activity="walking", count=3)
            ("How many walking bouts occurred?", "There are 3 walking bouts.")
        """
        variants = self.templates.get(task)
        if not variants:
            raise ValueError(f"No templates registered for task: {task}")

        # Randomly select a template variant
        variant = variants[rng.integers(0, len(variants))]

        # Auto-generate grammatical helpers based on provided kwargs
        filled_kwargs = self._add_grammar_helpers(kwargs)

        # Fill placeholders
        try:
            question = variant.question.format(**filled_kwargs)
            answer = variant.answer.format(**filled_kwargs)
        except KeyError as e:
            raise ValueError(
                f"Missing placeholder {e} for task '{task}'. Provided: {list(kwargs.keys())}"
            )

        # Apply grammar corrections
        question = self._check_a_an(question)
        answer = self._check_a_an(answer)

        return question, answer

    def _add_grammar_helpers(self, kwargs: Dict) -> Dict:
        """
        Auto-generate grammatical helper variables based on context.

        Handles:
        - Singular/plural: is/are, bout/bouts, time/times, episode/episodes
        - Boolean: yes/no, does/doesn't, present/absent
        - Ordinals: first, second, third
        - Ordering: came before/after
        """
        filled = dict(kwargs)

        # Count-based singular/plural
        if "count" in filled:
            count = filled["count"]
            filled["is_are"] = "is" if count == 1 else "are"
            filled["bout_bouts"] = "bout" if count == 1 else "bouts"
            filled["time_times"] = "time" if count == 1 else "times"
            filled["episode_episodes"] = "episode" if count == 1 else "episodes"
            filled["event_events"] = "event" if count == 1 else "events"
            filled["instance_instances"] = "instance" if count == 1 else "instances"
            filled["period_periods"] = "period" if count == 1 else "periods"
            filled["occurrence_occurrences"] = "occurrence" if count == 1 else "occurrences"

        # Boolean yes/no helpers
        if "exists" in filled:
            exists = filled["exists"]
            filled["yes_no"] = "Yes" if exists else "No"
            filled["does_doesnt"] = "does" if exists else "doesn't"
            filled["present_absent"] = "present" if exists else "absent"

        # Anomaly detection/localization helpers
        if "is_anomaly" in filled:
            is_anomaly = filled["is_anomaly"]
            filled["yes_no"] = "Yes" if is_anomaly else "No"

        # Ordering helpers
        if "a_before_b" in filled:
            a_before_b = filled["a_before_b"]
            filled["yes_no"] = "Yes" if a_before_b else "No"
            filled["came_before_after"] = "came before" if a_before_b else "came after"

        # Ordinal conversion
        if "k" in filled:
            k = filled["k"]
            filled["ordinal"] = self.ORDINALS.get(k, f"{k}th")

        # Duration formatting (if duration_ms provided)
        if "duration_ms" in filled:
            duration_ms = filled["duration_ms"]
            if duration_ms >= 60000:
                mins = duration_ms / 60000
                filled["duration"] = f"{mins:.1f} minutes"
            else:
                secs = duration_ms / 1000
                filled["duration"] = f"{secs:.1f} seconds"

        return filled

    @staticmethod
    def _check_a_an(sentence: str) -> str:
        """
        Auto-correct a/an based on following word's initial sound.

        This prevents grammatical errors like "a accelerometer" -> "an accelerometer".
        Inspired by SensorLM's grammar correction approach.
        """
        # Pattern to find 'a' or 'an' followed by a word
        pattern = r"\b(a|an|A|An)\s+(\w+)"

        def replace_func(match):
            article = match.group(1)
            next_word = match.group(2)
            vowels = "aeiouAEIOU"

            # Determine correct article based on following word
            should_be_an = next_word[0] in vowels

            # Handle special cases (e.g., "hour" starts with 'h' but sounds like vowel)
            # For simplicity, we only handle the basic vowel case
            if article.lower() == "a" and should_be_an:
                new_article = "An" if article[0].isupper() else "an"
                return f"{new_article} {next_word}"
            elif article.lower() == "an" and not should_be_an:
                new_article = "A" if article[0].isupper() else "a"
                return f"{new_article} {next_word}"

            return match.group(0)  # No change needed

        return re.sub(pattern, replace_func, sentence)

    def get_template_count(self, task: str) -> int:
        """Return number of template variants for a task."""
        return len(self.templates.get(task, []))

    def register_templates(self, task: str, variants: List[TemplateVariant]) -> None:
        """Add new templates for a task at runtime."""
        self.templates.setdefault(task, []).extend(variants)

    def get_available_tasks(self) -> List[str]:
        """Return list of tasks with registered templates."""
        return list(self.templates.keys())

    def get_all_template_counts(self) -> Dict[str, int]:
        """Return template counts for all tasks."""
        return {task: len(variants) for task, variants in self.templates.items()}

    @classmethod
    def from_json(cls, path: str) -> "PromptTemplateBank":
        """
        Load templates from a JSON file.

        Expected format:
        {
            "existence": [
                {"question": "Is there {activity}?", "answer": "{yes_no}"},
                ...
            ],
            ...
        }
        """
        with open(path) as f:
            data = json.load(f)

        custom = {}
        for task, variants in data.items():
            custom[task] = [TemplateVariant(**v) for v in variants]

        return cls(custom_templates=custom)
