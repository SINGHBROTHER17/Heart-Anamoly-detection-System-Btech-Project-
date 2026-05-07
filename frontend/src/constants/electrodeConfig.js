/**
 * Electrode configuration — all metadata for the 10 ECG electrodes and the
 * 9-step recording sequence.
 *
 * Coordinate system: SVG viewBox "0 0 200 330" (full body front view, anatomically
 * correct — patient's RIGHT appears on the LEFT side of the diagram).
 */

export const ELECTRODES = {
  RA: {
    label: 'RA',
    fullName: 'Right Arm',
    color: '#E53E3E',       // red
    svgX: 8,
    svgY: 92,
    instruction: 'Attach the RED clip to your inner right wrist or lower right forearm, bony side facing down.',
    tip: 'Place near the wrist bone — bone-adjacent placement reduces muscle noise.',
    commonMistake: "Don't place on the palm side of the wrist or over a tendon.",
    difficulty: 'Easy',
    group: 'limb',
  },
  LA: {
    label: 'LA',
    fullName: 'Left Arm',
    color: '#D4A017',       // amber (readable on skin)
    svgX: 192,
    svgY: 92,
    instruction: 'Attach the YELLOW clip to your inner left wrist or lower left forearm, bony side facing down.',
    tip: 'Mirror the right arm position exactly for consistent readings.',
    commonMistake: "Avoid placing over the pulse point — this adds artifact.",
    difficulty: 'Easy',
    group: 'limb',
  },
  RL: {
    label: 'RL',
    fullName: 'Right Leg',
    color: '#4A5568',       // dark grey (ground)
    svgX: 48,
    svgY: 304,
    instruction: 'Attach the BLACK (ground) clip to your right ankle or just above the ankle bone.',
    tip: 'This is the ground electrode — placement is less critical than chest leads.',
    commonMistake: "Don't place over scar tissue or hair if you can avoid it.",
    difficulty: 'Easy',
    group: 'limb',
  },
  LL: {
    label: 'LL',
    fullName: 'Left Leg',
    color: '#38A169',       // green
    svgX: 152,
    svgY: 304,
    instruction: 'Attach the GREEN clip to your left ankle or just above the left ankle bone.',
    tip: 'Keep it symmetric with the right leg electrode for balanced recording.',
    commonMistake: "Avoid very bony prominences — the flat side of the ankle is better.",
    difficulty: 'Easy',
    group: 'limb',
  },
  V1: {
    label: 'V1',
    fullName: 'Chest Lead V1',
    color: '#F6AD55',       // orange
    svgX: 90,
    svgY: 126,
    instruction: 'Place on the RIGHT side of your breastbone (sternum), at the level of your 4th rib space.',
    tip: 'Feel for your collarbone, count down 4 rib gaps — V1 is just to the right of the sternum at that level.',
    commonMistake: "Don't place directly on the breastbone itself — it must be to the right of it.",
    difficulty: 'Moderate',
    group: 'chest',
    ribCount: 4,
    side: 'right of sternum',
  },
  V2: {
    label: 'V2',
    fullName: 'Chest Lead V2',
    color: '#68D391',       // light green
    svgX: 110,
    svgY: 126,
    instruction: 'Place on the LEFT side of your breastbone (sternum), at the exact same rib level as V1.',
    tip: 'V1 and V2 should be at the same height, one on each side of the sternum.',
    commonMistake: "Don't place V2 lower than V1 — they must be at the same rib level.",
    difficulty: 'Moderate',
    group: 'chest',
    ribCount: 4,
    side: 'left of sternum',
  },
  V3: {
    label: 'V3',
    fullName: 'Chest Lead V3',
    color: '#76E4F7',       // cyan
    svgX: 116,
    svgY: 142,
    instruction: 'Place halfway between V2 and V4 — diagonally across the chest between those two electrodes.',
    tip: 'There is no fixed rib landmark for V3 — find V2 and V4 first, then split the distance.',
    commonMistake: "Don't guess — if V2 and V4 are placed correctly, V3 position is unambiguous.",
    difficulty: 'Moderate',
    group: 'chest',
    side: 'between V2 and V4',
  },
  V4: {
    label: 'V4',
    fullName: 'Chest Lead V4',
    color: '#B794F4',       // purple
    svgX: 122,
    svgY: 158,
    instruction: "Find your left nipple line (the vertical line through the nipple). Place V4 at the 5th rib space along that line.",
    tip: "For women: the midclavicular line is midway between the center of the shoulder and the sternum.",
    commonMistake: "Don't follow the nipple exactly in all cases — use the midclavicular line as the reference.",
    difficulty: 'Moderate',
    group: 'chest',
    ribCount: 5,
    side: 'left midclavicular',
  },
  V5: {
    label: 'V5',
    fullName: 'Chest Lead V5',
    color: '#F687B3',       // pink
    svgX: 138,
    svgY: 158,
    instruction: 'Place at the SAME HEIGHT as V4, but moved toward the left armpit — at the anterior axillary line.',
    tip: 'The anterior axillary line runs along the front edge of your armpit.',
    commonMistake: "Don't move V5 lower — keep it at exactly the same vertical level as V4.",
    difficulty: 'Tricky',
    group: 'chest',
    side: 'anterior axillary',
  },
  V6: {
    label: 'V6',
    fullName: 'Chest Lead V6',
    color: '#90CDF4',       // light blue
    svgX: 156,
    svgY: 157,
    instruction: 'Place at the SAME HEIGHT as V4 and V5, now at the side of the chest directly below the armpit (midaxillary line).',
    tip: 'Run your finger straight down from the center of your armpit — that vertical line is the midaxillary line.',
    commonMistake: "Don't go too far back — V6 should still be on the side, not the back, of the chest.",
    difficulty: 'Tricky',
    group: 'chest',
    side: 'midaxillary',
  },
  WST: {
    label: 'WST',
    fullName: 'Lower Left Abdomen',
    color: '#9F7AEA',       // violet
    svgX: 76,
    svgY: 244,              // lower-left abdomen on the body
    instruction: 'Place the third electrode at the bottom of your LEFT waist, just below the rib cage and to the left of the navel.',
    tip: 'Aim for a flat patch of skin below the costal margin — this captures the Lead II reading.',
    commonMistake: "Don't place it too low (over the hip bone) or directly on the navel.",
    difficulty: 'Easy',
    group: 'limb',
  },
};

/**
 * Recording sequence: V1 → V6 first (chest leads), then L2 then L1
 * (limb leads using a 3-electrode setup: RA, LA, and a moving Waist electrode).
 *
 *   L2: RA + LA + WST  (Lead II — measures right arm to lower-left waist)
 *   L1: RA + LA        (Lead I  — measures right arm to left arm; waist removed)
 */
export const LEAD_SEQUENCE = [
  {
    id: 'V1',
    stepLabel: 'V1',
    leads: ['V1'],
    electrodes: ['V1'],
    duration: 15,
    title: 'Step 1 — Chest Lead V1',
    shortDesc: 'Right of sternum, 4th rib space',
  },
  {
    id: 'V2',
    stepLabel: 'V2',
    leads: ['V2'],
    electrodes: ['V2'],
    duration: 15,
    title: 'Step 2 — Chest Lead V2',
    shortDesc: 'Left of sternum, same level as V1',
  },
  {
    id: 'V3',
    stepLabel: 'V3',
    leads: ['V3'],
    electrodes: ['V3'],
    duration: 15,
    title: 'Step 3 — Chest Lead V3',
    shortDesc: 'Between V2 and V4',
  },
  {
    id: 'V4',
    stepLabel: 'V4',
    leads: ['V4'],
    electrodes: ['V4'],
    duration: 15,
    title: 'Step 4 — Chest Lead V4',
    shortDesc: '5th rib, midclavicular line',
  },
  {
    id: 'V5',
    stepLabel: 'V5',
    leads: ['V5'],
    electrodes: ['V5'],
    duration: 15,
    title: 'Step 5 — Chest Lead V5',
    shortDesc: 'Same height as V4, anterior axillary line',
  },
  {
    id: 'V6',
    stepLabel: 'V6',
    leads: ['V6'],
    electrodes: ['V6'],
    duration: 15,
    title: 'Step 6 — Chest Lead V6',
    shortDesc: 'Same height as V4, midaxillary line',
  },
  {
    id: 'L2',
    stepLabel: 'L2',
    leads: ['II'],
    electrodes: ['RA', 'LA', 'WST'],
    duration: 15,
    title: 'Step 7 — Lead II (L2)',
    shortDesc: 'RA + LA stay attached; place 3rd electrode on lower-left waist',
  },
  {
    id: 'L1',
    stepLabel: 'L1',
    leads: ['I'],
    // Waist electrode removed for L1 — only RA and LA remain attached.
    electrodes: ['RA', 'LA'],
    duration: 15,
    title: 'Step 8 — Lead I (L1)',
    shortDesc: 'Remove waist electrode; record between RA and LA',
  },
  {
    id: 'confirm',
    stepLabel: 'Confirm',
    leads: [],
    electrodes: [],
    duration: 0,
    title: 'Step 9 — Review & Confirm',
    shortDesc: 'Check signal quality for all leads',
  },
];

/** Duration of each non-confirm step in seconds. */
export const RECORDING_DURATION_S = 15;

/** Sample rate sent to backend. */
export const SAMPLE_RATE = 500;
