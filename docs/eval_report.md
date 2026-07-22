# Evaluation Report

Eval set: `docs/eval_set.json` (8 questions)

**Ground truth is source-file-level** (did the correct document surface in the
top-K retrieved chunks), not chunk-level - a coarser but honest signal given
this eval set doesn't have per-chunk relevance annotations yet. Includes
multi-part, lexical-overlap, and negative-control questions specifically
designed to differentiate naive from advanced retrieval.

## Summary

| Config | Precision@5 | Recall@5 | MRR |
|---|---|---|---|
| Naive (single-pass, no rerank/grade/compress) | 0.857 | 0.857 | 0.857 |
| Advanced (full pipeline) | 0.893 | 1.0 | 0.905 |

**Hallucination check:** 0/1 negative-control question(s) answered honestly (naive), 0/1 (advanced).

## Per-question detail

### q1: What type of grease is recommended for the spindle of Horizontal SCARA robots, and after how many hours of movement should the customer apply it?
- Expected source: `ASDAVS-SP002A_Robot_Periodic_Maintenance_Specification_July_2023.pdf`
- Naive retrieved: ['ASDAVS-SP002A_Robot_Periodic_Maintenance_Specification_July_2023.pdf', 'ASDAVS-SP002A_Robot_Periodic_Maintenance_Specification_July_2023.pdf'] (MRR: 1.0)
- Advanced retrieved: ['ASDAVS-SP002A_Robot_Periodic_Maintenance_Specification_July_2023.pdf', 'ASDAVS-SP002A_Robot_Periodic_Maintenance_Specification_July_2023.pdf', 'ASDAVS-SP002A_Robot_Periodic_Maintenance_Specification_July_2023.pdf', 'ASDAVS-SP002A_Robot_Periodic_Maintenance_Specification_July_2023.pdf'] (MRR: 1.0)
- Naive keyword coverage: 67%
- Advanced keyword coverage: 100%

### q2: Which FANUC CNC series is this Operation and Maintenance Handbook designed for?
- Expected source: `A16B-2200-0900.pdf`
- Naive retrieved: ['A16B-2200-0900.pdf', 'A16B-2200-0900.pdf'] (MRR: 1.0)
- Advanced retrieved: ['A16B-2200-0900.pdf', 'A16B-2200-0900.pdf', 'A16B-2200-0900.pdf', 'A16B-2200-0900.pdf'] (MRR: 1.0)
- Naive keyword coverage: 100%
- Advanced keyword coverage: 0%

### q3: What happens when the smartPAD is disconnected or removed from the system layout?
- Expected source: `KUKA_Sunrise_Cabinet_Med_en.pdf`
- Naive retrieved: ['KUKA_Sunrise_Cabinet_Med_en.pdf', 'KUKA_Sunrise_Cabinet_Med_en.pdf'] (MRR: 1.0)
- Advanced retrieved: ['KUKA_Sunrise_Cabinet_Med_en.pdf', 'KUKA_Sunrise_Cabinet_Med_en.pdf', 'KUKA_Sunrise_Cabinet_Med_en.pdf', 'KUKA_Sunrise_Cabinet_Med_en.pdf'] (MRR: 1.0)
- Naive keyword coverage: 0%
- Advanced keyword coverage: 100%

### q4: What safety stop is triggered by fully pressing the enabling switch to the panic position?
- Expected source: `KUKA_Sunrise_Cabinet_Med_en.pdf`
- Naive retrieved: ['KUKA_Sunrise_Cabinet_Med_en.pdf', 'KUKA_Sunrise_Cabinet_Med_en.pdf'] (MRR: 1.0)
- Advanced retrieved: ['KUKA_Sunrise_Cabinet_Med_en.pdf', 'KUKA_Sunrise_Cabinet_Med_en.pdf', 'KUKA_Sunrise_Cabinet_Med_en.pdf'] (MRR: 1.0)
- Naive keyword coverage: 50%
- Advanced keyword coverage: 50%

### q5_multipart: What grease specification applies to Horizontal SCARA robots, and separately, what safety stop is triggered by the enabling switch in panic position?
- Expected source: `ASDAVS-SP002A_Robot_Periodic_Maintenance_Specification_July_2023.pdf`
- Naive retrieved: ['KUKA_Sunrise_Cabinet_Med_en.pdf', 'KUKA_Sunrise_Cabinet_Med_en.pdf'] (MRR: 0.0)
- Advanced retrieved: ['KUKA_Sunrise_Cabinet_Med_en.pdf', 'KUKA_Sunrise_Cabinet_Med_en.pdf', 'ASDAVS-SP002A_Robot_Periodic_Maintenance_Specification_July_2023.pdf', 'KUKA_Sunrise_Cabinet_Med_en.pdf'] (MRR: 0.3333333333333333)
- Naive keyword coverage: 67%
- Advanced keyword coverage: 67%

### q6_lexical_overlap: What operating modes are supported by the robot controller, and what do T1 and T2 mean?
- Expected source: `KUKA_Sunrise_Cabinet_Med_en.pdf`
- Naive retrieved: ['KUKA_Sunrise_Cabinet_Med_en.pdf', 'KUKA_Sunrise_Cabinet_Med_en.pdf'] (MRR: 1.0)
- Advanced retrieved: ['KUKA_Sunrise_Cabinet_Med_en.pdf', 'KUKA_Sunrise_Cabinet_Med_en.pdf', 'KUKA_Sunrise_Cabinet_Med_en.pdf', 'KUKA_Sunrise_Cabinet_Med_en.pdf'] (MRR: 1.0)
- Naive keyword coverage: 100%
- Advanced keyword coverage: 100%

### q7_narrow_detail: What is the workspace, danger zone, and safety zone of a robot, as defined in the safety documentation?
- Expected source: `KUKA_Sunrise_Cabinet_Med_en.pdf`
- Naive retrieved: ['KUKA_Sunrise_Cabinet_Med_en.pdf', 'KUKA_Sunrise_Cabinet_Med_en.pdf'] (MRR: 1.0)
- Advanced retrieved: ['KUKA_Sunrise_Cabinet_Med_en.pdf', 'KUKA_Sunrise_Cabinet_Med_en.pdf', 'KUKA_Sunrise_Cabinet_Med_en.pdf'] (MRR: 1.0)
- Naive keyword coverage: 100%
- Advanced keyword coverage: 100%

### q8_negative_control: What is the maximum flight altitude certified for this robot's drone attachment?
- Type: negative control (no correct source in corpus)
- Naive honest decline: False
- Advanced honest decline: False
- Naive reply preview: 'A maximum flight altitude is not stated in the document chunks.\n\nReference:\n- KUKA_Sunrise_Cabinet_Med_en.pdf'
- Advanced reply preview: '60/114\n\nReference:\n- KUKA_Sunrise_Cabinet_Med_en.pdf'
