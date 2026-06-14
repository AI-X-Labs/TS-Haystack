# SPDX-License-Identifier: CC-BY-NC-4.0

"""
Sleep PSG Annotation-Based Retrieval Benchmark.

A retrieval benchmark for polysomnography data using natural annotations.
Uses full unmodified recordings and generates questions from real sleep
stage and arousal event annotations.

Two datasets from the PhysioNet 2018 Challenge (13-channel PSG, 200Hz):
- Sleep stages (7 tasks): Wake, N1, N2, N3, REM
- Arousals (6 tasks): rera, hypopnea, obstructive_apnea, central_apnea, mixed_apnea
"""
